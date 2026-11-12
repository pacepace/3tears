# 3tears Integration Guide

This guide wires 3tears into a host application as its data layer: the mental model,
the decisions you have to make, and copy-pasteable "hello world" steps for the L1 / L2 /
L3 tiers, for local development and for a multi-pod deployment. It assumes no particular
host app, cloud, or platform. Where a sample and the code disagree, the code is right
and the guide has a bug. The known gaps are in
[§13](#13-known-gaps).

The core follows the family's rule: a package requires as little of its host as it can,
and everything past that is optional. Here the only requirement is an L3; L2, the proxy,
leases and buffered flushing are all opt-in. The [root README](../README.md#adopting-a-package)
states the rule, and [Evals architecture](evals-architecture.md) shows it for a package
whose ports a host can fill with this data layer or with its own.

---

## Table of contents

1. [What 3tears is (and is not)](#1-what-3tears-is-and-is-not)
2. [The core mental model](#2-the-core-mental-model)
3. [The three tiers at a glance](#3-the-three-tiers-at-a-glance)
4. [The building blocks](#4-the-building-blocks)
5. [Design principles](#5-design-principles)
6. [Decisions you have to make](#6-decisions-you-have-to-make)
7. [Prerequisites & installation](#7-prerequisites--installation)
8. [Hello world](#8-hello-world)
9. [Schema & migrations](#9-schema--migrations)
10. [Configuration reference](#10-configuration-reference)
11. [Observability hookup](#11-observability-hookup)
12. [Deployment considerations](#12-deployment-considerations)
13. [Known gaps](#13-known-gaps)
14. [Integration checklist](#14-integration-checklist)
15. [Appendix: import cheat-sheet](#15-appendix-import-cheat-sheet)

---

## 1. What 3tears is (and is not)

3tears is a **Python library suite**, not a service.

- There is **no daemon, container, sidecar, or API** to deploy for 3tears itself.
- You `pip install` the packages you need and construct objects **inside your host
  application's process**.
- It brings **infrastructure dependencies** (PostgreSQL or a git working tree for L3,
  optionally a NATS server), but *those* are services you stand up and own -- 3tears
  just talks to them.

The monorepo publishes many packages, each installed on its own and all released at one
version (the full table is in the [root README](../README.md#packages)). The ones that
matter for the data layer:

| Package (PyPI) | Import root | Role in the data layer |
|---|---|---|
| `3tears` | `threetears.core` | Three-tier entities/collections, L1 cache, `DataStore`, schema, the canonical `MigrationRunner`, coordination (leases), security sandbox |
| `3tears-nats` | `threetears.nats` | The **L2** client: canonical `NatsClient`, `Subjects`, JetStream KV bucket helper, typed pub/sub |
| `3tears-observe` | `threetears.observe` | Structured logging, OpenTelemetry `@traced`, `set_context` |
| `3tears-agent-memory` | `threetears.agent.memory` | LLM agent memory: extraction, retrieval, hybrid search (needs pgvector) |

This guide focuses on **`threetears.core`** plus **`threetears.nats`** for L2.

---

## 2. The core mental model

> **The library instruments; the host configures.**

3tears never opens a database connection, reads an environment variable for config,
or installs a log handler on your behalf. **You** construct the backing clients (a
PostgreSQL pool, optionally a `NatsClient`), hand them to 3tears, and **you own
their lifecycle** (create on startup, close on shutdown).

All three tiers funnel through one dependency-injection seam -- the
**`CollectionRegistry`**. You `configure(l1_backend=, l2_client=, kv_key_scope=,
l3_pool=)` once; every collection (including those `DataStore` builds for you)
resolves all three tiers from the registry. `kv_key_scope` is required wherever an
`l2_client` is -- see §8.2. A collection's L2 client can still be overridden per
construction (an explicit `nats_client=` argument wins, and an explicit `None`
disables L2 for that collection regardless of the registry), but the registry is
the default wiring path -- see §4.

```text
        ┌─────────────────────── your host app process ───────────────────────┐
        │                                                                      │
        │   you create:   SQLiteBackend       asyncpg pool      NatsClient*    │
        │                      │                   │                 │         │
        │                      ▼                   ▼                 ▼         │
        │  CollectionRegistry.configure(l1_backend=, l3_pool=,                 │
        │                               l2_client=, kv_key_scope=)             │
        │                      │                                               │
        │                      ▼                                               │
        │   DataStore(namespace_id, registry) ─► collections ─► entities       │
        │                                                                      │
        └──────────────────────────────────────────────────────────────────────┘
                * NATS / L2 is optional -- single-pod apps skip it (§3, §8.2).
```

---

## 3. The three tiers at a glance

A read walks L1 → L2 → L3 and **re-promotes** on the way back. A write goes to L3
(source of truth), promotes into L1 and L2, and broadcasts a cross-pod invalidation.
A miss at any tier falls through to the next -- **the stack degrades gracefully**.

| Tier | Implementation | Nature | Source of truth? | Required? |
|---|---|---|---|---|
| **L1** | `SQLiteBackend` (`threetears.core.cache.sqlite`) | **In-process, in-memory** SQLite (named `memdb` VFS). Fast local cache, **not durable** -- rebuilt from L3 on restart. | No | Practically yes (collections cache through it) |
| **L2** | `NatsClient` JetStream KV (`threetears.nats`) | **Cross-pod** shared cache + typed cache-coherence pub/sub. Bucket is `{namespace}-collections`, on the storage its declarer chooses; declare it on **file** storage so it survives a broker restart (§12, Step 3). | No | **No** -- only matters with >1 pod |
| **L3** | an asyncpg-style pool, `NatsProxyL3Backend`, or a git working tree | **Durable storage. The source of truth.** | **Yes** | **Yes, always** |

Consequences:

- **L1 holds nothing durable.** Restart the process and L1 is empty; never treat it
  as a store.
- **Single pod? Skip L2.** With one process there is no cross-pod cache to keep
  coherent. Run L1 + L3 and add NATS when you scale out.
- **L3 is the only thing you cannot omit.**

### Three ways to reach L3

1. **Direct pool** -- the process holds DB credentials and you pass a connection pool.
   The pool is **duck-typed**: any object with `async` `.execute()` / `.fetch()` /
   `.fetchrow()` works; `asyncpg` pools satisfy this directly.
2. **Proxy (`NatsProxyL3Backend`)** -- the process has **no DB credentials**. It
   serializes SQL and sends it over NATS request/reply to a central broker that owns
   the real pool, on the subjects below. Use for sandboxed worker pods. **Note:** only
   the *client* ships in `threetears.core`; the broker that answers those subjects is
   **not** part of the package (see §13).
3. **A git working tree** (`threetears.core.backends.git`, with the `nats` extra) -- rows
   are files and every change set is one commit, with one writer per repository held by a
   NATS lease. [core-git.md](adoption/core-git.md) covers when to choose it; the rest of
   this guide uses a PostgreSQL L3.

### The L3 broker's subjects

A broker answers every one of these on `<ns>.` (the client's `namespace_prefix`), or
the client calls that need it raise `DataLayerUnavailableError`. Each subject's helper
is `threetears.nats.subjects.Subjects.l3_*`; the request bodies are pinned field for
field in `packages/core/tests/unit/backends/` and stated in the CHANGELOG's "the wire"
entries.

| Subject | Sent by | Contract |
|---|---|---|
| `l3.query` | every read and single write | one statement, its own transaction; replies rows or a row count, plus a write's `generations` |
| `l3.batch` | `NatsProxyL3Backend.execute_batch` | several statements in one request, in one transaction when it asks for one |
| `l3.tx.begin` / `execute` / `fetchrow` / `fetch` / `commit` / `rollback` | a transaction opened on the proxy (`transaction()`) | one explicit transaction the broker holds open across requests, by `tx_id` |
| `l3.tx.copy` | `save_rows(..., copy=True)` inside a transaction | body `{correlation_id, identity_token, tx_id, table, columns, rows}`, each value encoded as a statement parameter (`transport_param`); insert-only `COPY` into the open transaction; a key already held fails it as `asyncpg.UniqueViolationError` |
| `l3.truncate` | `SchemaBackedCollection.truncate()` | body `{correlation_id, identity_token, namespace, table, timeout_ms}`; `TRUNCATE` alone, outside any transaction, under the database's DDL lock, only for a declared table of the caller's own data space; replies as `l3.query` does |
| `l3.load_files` | `SchemaBackedCollection.load_files()` | body `{correlation_id, identity_token, namespace, table, columns, constants, files: [{url, rows}], format: "parquet", rows_per_transaction, timeout_ms}`; the broker fetches each file and `COPY`s its rows into a declared table of the caller's own data space, committing every `rows_per_transaction` rows, outside any transaction; replies `{success, row_count, files: [{index, rows_loaded}]}` plus `generations`. It fetches only a presigned `https` URL (no user info, no port but 443, no fragment, no redirect followed) on an S3 host of the caller's export bucket, under the prefix of an export recorded for that caller, and refuses the whole request otherwise (`LoadingStore`'s docstring). One request's wait (`timeout_ms` per transaction) never passes `LOAD_FILES_MAX_WAIT_SECONDS`: `load_files()` sends a longer load as several requests of whole files, in order (`load_requests`), and the proxy refuses a request that would pass it |

A broker that predates a subject answers it with an error, so upgrade the broker before
any pod sends `l3.truncate`, `l3.tx.copy` or `l3.load_files`.

---

## 4. The building blocks

Unless noted, these are exported from `threetears.core` (see §15).

- **`CollectionRegistry`** -- DI container + table-name lookup + cache coherence.
  Holds default L1/L2/L3 backends; `configure(l1_backend=, l2_client=, kv_key_scope=,
  l3_pool=)` sets defaults -- `kv_key_scope` is required alongside any `l2_client` -- 
  and `bind_table(table_name, l1_backend=, l2_client=, l3_pool=)` pins
  per-table overrides **before** a collection is constructed (needed because a
  collection snaps its backends from the registry at `__init__`). A collection
  resolves its L2 client from the registry (`get_l2_client(table_name)`) when no
  `nats_client=` constructor argument is supplied; an explicit argument wins, and an
  explicit `None` disables L2 for that collection. Drives cross-pod invalidation via
  `start_invalidation_listener(nats)` / `publish_invalidation(...)` using a typed
  `CacheInvalidationMessage` on `Subjects.cache_invalidate()`.
  `stop_invalidation_listener()` is the teardown half: it unsubscribes the listener and
  is safe to call from a `finally` -- a second call, or one with nothing subscribed, is a
  no-op, and a stop on an already-draining connection does not raise. A second
  `start_invalidation_listener` while one is live is likewise a no-op rather than a
  second consumer, so the registry is reusable across a stop/start.
- **`DataStore`** -- the ergonomic front door. Wraps a registry, creates tables from
  declarative definitions, hands you collections by name (`store["my_table"]`),
  exposes raw `query` / `execute`, and `run_migrations(runner)`. `create_table` runs its
  DDL under the database-wide DDL lock; `ddl_session()` yields the store pinned to one
  connection holding that lock, for DDL of your own.
- **`BaseEntity`** -- a change-tracking record (`entity.field = x` marks it dirty)
  with optimistic concurrency (`date_updated` mismatch → `ConcurrentModificationError`).
  `await entity.save()` / `reload()`; `.id`, `.is_new`.
- **`BaseCollection`** -- the per-table gateway implementing the three-tier logic.
  **Composite primary keys are first-class** (`primary_key_column = ("a", "b")`).
  Its `l3_pool` attribute is public -- drop to `await self.l3_pool.fetch(...)` for
  ad-hoc SQL the collection API can't express. You rarely subclass it by hand -- 
  `DataStore` generates one per table.
- **Schema definitions** -- `TableDef`, `ColumnDef`, `IndexDef`, `ForeignKeyDef`
  (validated Pydantic models).
- **Migrations** (`threetears.core.data.migrations`) -- the canonical
  `MigrationRunner`, `PackageMigrations`, `MigrationScope` (see §9).
- **`CoreConfig` / `DefaultCoreConfig`** -- flush-strategy and caching config (a
  `Protocol`, so your own settings object can satisfy it without inheritance).
- **Coordination** -- `KVLease` + `LeaseHandle` (NATS-backed distributed lock) for
  multi-pod mutual exclusion.

---

## 5. Design principles

1. **Library instruments, host configures.** You own connections, lifecycle,
   secrets, logging destinations, and trace exporters.
2. **Minimal required, maximal optional.** An L3 is the one requirement; every other
   tier and feature is opt-in, so a single pod starts with L1 + L3 and adds NATS later.
3. **L3 is the source of truth; L1/L2 are disposable.** Losing all caches is a
   performance event, not a correctness event.
4. **Graceful degradation.** A missing/unhealthy cache tier falls through to the
   next. The L2 path narrows its exception scope to real transport errors (`KvError`)
   and degrades to L1+L3; programming errors still surface.
5. **Explicit dependency injection.** One registry, configured once; all three tiers
   resolve through it. (A collection may still override its L2 client at construction;
   see §4.)
6. **Caller owns concurrency boundaries.** Optimistic locking raises on conflict; the
   host decides retry/merge policy. `KVLease` is available for cross-pod mutual
   exclusion.
7. **Schema is declarative and validated.** Identifiers must match
   `^[a-z][a-z0-9_]*$`; column types are a fixed allowed set (§9).
8. **One blessed migration path.** No alembic, no autogen, no back-compat shims -- 
   hand-written, idempotent, version-tracked migrations (§9).

---

## 6. Decisions you have to make

| Decision | Options | Guidance |
|---|---|---|
| **Single pod or multiple?** | one / many | One → **skip L2/NATS**. Many → add NATS for L2 and cross-pod invalidation. |
| **How does the process reach L3?** | direct pool / NATS proxy / git working tree | Trusted service with DB creds → direct pool. Sandboxed worker without creds → proxy (and you must provide the broker -- §13). Rows that belong in a repository → git ([core-git.md](adoption/core-git.md)). |
| **How are tables defined?** | `DataStore` dynamic / hand-written `BaseCollection` | Start with `DataStore` -- its collections inherit all three tiers (incl. L2) from the registry. Drop to hand-written subclasses for bespoke serialization or composite-PK control. |
| **Which packages?** | `3tears` / `3tears[nats]` / + `agent-memory` | KV/relational caching → `3tears` (`3tears[nats]` for L2 or a git L3). Semantic memory/search → add `agent-memory` and **enable pgvector**. |
| **Flush strategy** | `ALWAYS`, `ON_CHECKPOINT`, `ON_SCHEDULE`, `ON_SHUTDOWN` | `ALWAYS` = write-through, simplest to reason about. Buffered strategies trade durability latency for throughput (§10). |

---

## 7. Prerequisites & installation

**Host runtime** -- Python **3.14+** and an `async` runtime; the data API is
`async`/`await` end to end.

**Backing services**

- **PostgreSQL** -- for a PostgreSQL L3, which the rest of this guide uses. For
  `agent-memory` (vector/semantic search): `CREATE EXTENSION vector;` (pgvector).
- **NATS with JetStream** -- for L2 (more than one pod), the L3 proxy, or a git L3.

**Install (into your host app's environment)**

```bash
pip install 3tears                  # core data layer, with asyncpg for L3
pip install "3tears[nats]"          # plus the NATS client, for L2 or a git L3
pip install "3tears[nats,security]" # plus signed requests, for the L3 proxy
# pip install 3tears-agent-memory   # agent memory (requires pgvector)
```

For local integration tests, 3tears ships reusable testcontainer fixtures:
`pip install "3tears[testing]"` and see §8.1.

> The repo's own `uv sync` / `./scripts/*` are for **developing the framework**, not
> for consuming it. As a host app you depend on the published packages.

---

## 8. Hello world

Three progressively richer wirings. Each is self-contained inside an `async` entry
point. CRUD is identical across all three -- only the wiring differs.

### 8.1 Minimal: L1 + L3 (single pod, no NATS)

The smallest *correct* configuration: an in-memory L1 cache in front of a durable
PostgreSQL L3. No NATS. This exact shape -- the `DataStore` dynamic path with a raw
`asyncpg` pool and a `uuid` primary key -- round-trips end to end (covered by an
asyncpg-backed integration test in `threetears.core`).

```python
import asyncio
import uuid

import asyncpg

from threetears.core import (
    CollectionRegistry, DataStore, DefaultCoreConfig,
    TableDef, ColumnDef, IndexDef,
)
from threetears.core.cache.sqlite import SQLiteBackend


async def main() -> None:
    # --- 1. Build the tier backends (you own these) ---
    l1 = SQLiteBackend(db_name="hello_world")               # L1: in-process, in-memory
    pg_pool = await asyncpg.create_pool(                    # L3: durable source of truth
        dsn="postgresql://user:pass@localhost:5432/appdb",
    )

    # --- 2. Configure the DI seam (no l2_client for single-pod) ---
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pg_pool)

    # --- 3. Open a DataStore. The first argument is the agent id, which names the
    #        schema agent_<id hex>; the pool or broker binds search_path to it (§13).
    #        On a plain pool, as here, tables land in the pool's default schema. ---
    store = DataStore(uuid.uuid4(), registry, DefaultCoreConfig(collection_flush="ALWAYS"))

    # --- 4. Declare a table. create_table() runs the DDL on L3 and
    #        initializes the matching L1 schema for you. ---
    await store.create_table(
        TableDef(
            name="widgets",
            columns=[
                ColumnDef(name="id", column_type="uuid", primary_key=True),
                ColumnDef(name="name", column_type="text", nullable=False),
                ColumnDef(name="score", column_type="integer"),
            ],
            indexes=[IndexDef(name="idx_widgets_name", columns=["name"])],
        )
    )

    widgets = store["widgets"]   # a ready-to-use collection

    # --- 5. CRUD through all configured tiers ---
    wid = str(uuid.uuid4())
    entity = widgets.create({"id": wid, "name": "Sprocket", "score": 42})
    await entity.save()                 # writes L3, promotes into L1

    got = await widgets.get(wid)        # L1 hit -> returns an entity (or None)
    assert got is not None
    got.score = 99                      # change-tracked
    await got.save()

    await widgets.invalidate_cache(wid) # drop from L1 (and L2 if configured)
    refreshed = await widgets.get(wid)  # L1 miss -> L3 -> re-promote
    assert refreshed is not None and refreshed.score == 99

    await widgets.delete(wid)           # removes from every tier

    await pg_pool.close()               # you own shutdown


asyncio.run(main())
```

What happened: `create_table` ran the `CREATE TABLE`/index DDL against L3 and
initialized L1 from the same `TableDef`. `create()` returns an unsaved, dirty
entity; `save()` persists and promotes; `get()` reads through the tiers. With
`collection_flush="ALWAYS"`, every `save()` is write-through.

**Local integration testing.** Rather than hand-running Postgres, reuse the shipped
fixtures from a `conftest.py`:

```python
pytest_plugins = ["threetears.core.testing.fixtures"]
# session-scoped fixtures: db_container -> asyncpg URL (pgvector/pgvector:pg16),
#                          nats_container -> nats:// URI (JetStream on).
# For a plain image: @pytest.mark.parametrize("db_image", ["postgres:16"], indirect=True)
```
They gate on Docker and `pytest.skip` cleanly when it's unavailable.

### 8.2 Add L2 (NATS) for multi-pod caching + coherence

When you run more than one pod, add NATS so pods share an L2 cache **and** a write in
one pod evicts the stale copy in others. The only change from §8.1 is connecting a
`NatsClient` and handing it to the registry: `DataStore` collections then pick up L2
automatically, and each pod subscribes once for cross-pod invalidation. Install the
client with `pip install "3tears[nats]"`.

```python
import os
import uuid

import asyncpg

from threetears.core import CollectionRegistry, DataStore, DefaultCoreConfig
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import bind_collections_bucket
from threetears.nats import NatsClient, Principal, kv_key_scope_for


async def wire_with_l2(pg_pool: asyncpg.Pool) -> None:
    l1 = SQLiteBackend(db_name="hello_world")
    # The agent this pod serves, from its own env. NOT a fresh uuid per process: the
    # key scope is the SHARING boundary, so replicas of one agent must derive the same
    # value, or each replica caches into a keyspace of its own.
    agent_id = uuid.UUID(os.environ["THREETEARS_AGENT_ID"])

    # --- Connect NATS. The namespace prefixes the shared KV bucket,
    #     "{namespace}-collections". ---
    nats = await NatsClient.connect(
        nats_url=os.environ["THREETEARS_NATS_URL"],   # e.g. nats://nats:4222
        nats_subject_namespace="myapp",
        client_name="myapp-pod-1",
    )

    # --- Bind the shared collections bucket BEFORE configuring L2. One identity
    #     declares the bucket and every other process binds it, so a bucket this
    #     process cannot use fails here, at wiring time, rather than on the first read. ---
    #     Something must have declared it first, with `ensure_kv_bucket` (§12, Step 3).
    await bind_collections_bucket(nats, component="myapp-pod")

    # --- Configure all three tiers. `kv_key_scope` is MANDATORY alongside `l2_client`:
    #     `configure` raises `L2ScopeNotConfiguredError` without it. Every L2 key is
    #     `{scope}.{table}.{body}`, and the scope is the principal this process
    #     authenticates to NATS as, so a process cannot read another principal's keys.
    #     Derive it with `kv_key_scope_for`, never a literal: the NATS grant is minted
    #     from the same call. `Principal.AGENT_POD` requires the AGENT id, never a pod
    #     id; principals with no per-instance identity (`Principal.HUB`) take none. ---
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l3_pool=pg_pool,
        l2_client=nats,
        kv_key_scope=kv_key_scope_for(Principal.AGENT_POD, agent_id=agent_id),
    )

    # Collections the store builds now read and write L1 -> L2 -> L3.
    store = DataStore(agent_id, registry, DefaultCoreConfig(collection_flush="ALWAYS"))

    # --- Cross-pod cache coherence: each pod subscribes once at startup, so a write
    #     elsewhere evicts this pod's stale L1 entry. ---
    await registry.start_invalidation_listener(nats)

    # ... create tables and use `store` exactly as in §8.1 ...

    await registry.stop_invalidation_listener()   # release the subscription first
    await nats.shutdown()                         # graceful drain on shutdown
```

> Need L2 on for some tables but off for others? Override per table with
> `registry.bind_table("widgets", l2_client=nats)` before the collection is built.
> There is no per-table `kv_key_scope`: the scope names the PRINCIPAL, one process is
> one principal, so it belongs on the registry -- and `bind_table` raises
> `L2ScopeNotConfiguredError` if the registry has none yet.
>
> **A table with no L2 by design** (nothing on another replica reads its rows by key
> through a collection -- a loader's layer tables, a pod's report tables): build its
> collection with `nats_client=NO_L2` (`from threetears.core.collections import NO_L2`).
> It runs L1+L3 only, whatever L2 client the registry offers, and says so once per
> table at INFO (`collection runs without L2 by design: table=...`).
>
> **`nats_client=None` is not that declaration.** It also forces L1+L3, but reads as a
> client that should have been there: a collection with no resolvable L2 client and no
> `NO_L2` logs a one-shot WARNING (`collection invalidation is silently disabled`) on
> its first write that cannot broadcast, so a real wiring gap stays visible. Use `None`
> only where a missing L2 is a gap you want reported.

### 8.3 L3 over a proxy (no DB credentials in the pod)

For sandboxed pods that must not hold DB credentials, swap the direct pool for the
NATS proxy backend. Collection code is unchanged -- only what you pass as `l3_pool`
differs.

```python
from threetears.core.backends.nats_proxy import NatsProxyL3Backend

l3 = NatsProxyL3Backend(
    nats_client=nats,            # a connected NATS client
    namespace_prefix="myapp",    # subjects become myapp.l3.query, myapp.l3.batch, ... (§3)
    agent_id=str(agent_id),      # the AGENT id, never a pod id: it names the namespace
)
registry.configure(l1_backend=l1, l3_pool=l3)
```

> The broker that *answers* `myapp.l3.*` (owns the real PostgreSQL pool, executes
> SQL, returns rows) is **not** included in `threetears.core`. You must provide it,
> answering every subject in §3's table. See §13.

---

## 9. Schema & migrations

### Declarative schema

Validated Pydantic models:

- **`ColumnDef`**: `name`, `column_type`, `nullable=True`, `default=None`,
  `primary_key=False`, `vector_dim=None`. Allowed `column_type`: `text`, `integer`,
  `bigint`, `boolean`, `timestamp`, `timestamptz`, `uuid`, `jsonb`, `decimal`,
  `bytea`, `vector`. `timestamp` is naive (`TIMESTAMP`); `timestamptz` is
  timezone-aware (`TIMESTAMPTZ`, stored aware-UTC), for datetimes that flow
  through a timezone-aware layer such as the datasource broker.
  A `vector` column **requires** `vector_dim` (a positive int, the pgvector
  dimension) and `vector_dim` is only valid on a `vector` column -- both are validated
  at construction. It renders `VECTOR(<dim>)` DDL, binds with a `::vector` cast on
  write, and reads back as `list[float]`, so vector columns work over a plain asyncpg
  pool without registering the pgvector codec. (Materialising a `vector` column needs
  `pgvector` installed and `CREATE EXTENSION vector` in the database.)
- **`IndexDef`**: `name`, `columns`, `unique=False`.
- **`ForeignKeyDef`**: `name`, `columns`, `references_table`, `references_columns`,
  `on_delete="CASCADE"`, `on_update="NO ACTION"` (actions: `CASCADE`, `SET NULL`,
  `RESTRICT`, `NO ACTION`).
- **`TableDef`**: `name`, `columns`, `indexes=[]`, `foreign_keys=[]`.

Identifiers must match `^[a-z][a-z0-9_]*$`; invalid definitions raise at construction.

### Versioned migrations (the canonical runner)

> **The authoritative reference is [`how-to-add-a-migration.md`](./how-to-add-a-migration.md).**
> This section is an integration-level summary; that doc is the blessed path and
> covers file layout, scoping, rules, and testing in full.

The migration system is **package-composing**, not per-package. You declare a
`PackageMigrations` (a name, a `MigrationScope`, optional `depends_on`), register
version-tagged async callables on it, register the package with a single
`MigrationRunner`, and apply. The runner topologically orders packages by
`depends_on`, applies each pending `(version, package)`, and records them in a
`_schema_migrations` table keyed by `(version, package)`.

```python
from threetears.core import DataStore, TableDef, ColumnDef
from threetears.core.data.migrations import (
    MigrationRunner, PackageMigrations, MigrationScope,
)

pkg = PackageMigrations(name="myapp", scope=MigrationScope.AGENT)  # or .PLATFORM

@pkg.version(1)
async def create_widgets(store: DataStore) -> None:
    await store.create_table(TableDef(
        name="widgets",
        columns=[
            ColumnDef(name="id", column_type="uuid", primary_key=True),
            ColumnDef(name="name", column_type="text", nullable=False),
        ],
    ))

@pkg.version(2)
async def add_email(store: DataStore) -> None:
    await store.execute("ALTER TABLE widgets ADD COLUMN IF NOT EXISTS email TEXT")

@pkg.downgrade(2)                      # optional inverse; required only to roll back v2
async def drop_email(store: DataStore) -> None:
    await store.execute("ALTER TABLE widgets DROP COLUMN IF EXISTS email")

runner = MigrationRunner()
runner.register(pkg)

# Apply (idempotent). The DataStore is bound to its schema; the L3 layer sets
# search_path, so migration bodies use unqualified table names. The runner pins the
# store to one pooled connection for the whole run.
applied = await runner.apply_for_agent_schema(store)   # or: await store.run_migrations(runner)
# holding one plain connection instead: runner.apply_for_agent_schema(ConnectionSession(conn))
# isolation for package-local tests: await runner.apply_package(store, "myapp")
```

**One DDL job per database at a time.** Every run that writes -- both apply scopes,
`apply_package`, `downgrade_for_scope`, `stamp_version` -- holds the database-wide DDL
lock (`threetears.core.data.migrations.database_ddl_lock`) from before it reads
`_schema_migrations` until after its last row. It is keyed on `current_database()`, not
the schema: on YugabyteDB two index builds in one database hang each other until the
catalog-version wait times out, whichever schemas they are in. The lock is polled
(`pg_try_advisory_lock`), never blocked on, and it is a session lock, so a run happens on
one connection: a pool-backed `DataStore` is pinned to a connection the runner acquires,
and every body receives a `DataStore` bound to it -- `execute`, `query` and `create_table`
all reach the session that holds the lock, and a body's `create_table` does not take it
twice. A caller holding one plain connection passes `ConnectionSession(conn)`, which refuses
a pool. `MigrationRunner(lock_policy=DdlLockPolicy(max_wait=...))` bounds the wait with
`DdlLockTimeoutError`. Outside a migration, `DataStore.create_table` takes the lock itself;
any other code that runs DDL in the same database -- a background index build, a table
created by hand -- must take it too (`store.ddl_session()` or `database_ddl_lock`), and code
that abandons a long DDL statement must end its backend with `pg_terminate_backend`, since
`pg_cancel_backend` does not stop a YugabyteDB index build.

Rules that matter (full list in the how-to): **idempotent DDL only**
(`IF NOT EXISTS` / `IF EXISTS`), **unqualified names for agent scope** (search_path
is set for you), **one migration body per file**, **never edit an applied
migration**, and **no autogen / no shims**.

---

## 10. Configuration reference

### `CoreConfig` (in-code configuration)

`CoreConfig` is a `runtime_checkable` `Protocol`; `DefaultCoreConfig` is the concrete
default. Any object exposing these three attributes satisfies it.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `collection_flush` | `str` | `"ON_CHECKPOINT"` | Write-buffer strategy: `ALWAYS`, `ON_CHECKPOINT`, `ON_SCHEDULE`, `ON_SHUTDOWN`. `ALWAYS` = write-through. |
| `collection_flush_interval` | `int` (s) | `30` | Interval for `ON_SCHEDULE` flushing. |
| `collection_flush_tables` | `str` | `"messages,token_usage_logs"` | Comma-separated tables the buffered strategy applies to. |

> Buffered flushing only applies to tables listed in `collection_flush_tables` **and**
> only when a `WriteBuffer` is wired; otherwise writes are immediate. For predictable
> durability while learning, use `DefaultCoreConfig(collection_flush="ALWAYS")`.

### Environment variables

3tears reads a small set of env vars as fallbacks/tuning; per "host configures," most
behavior is set in code:

| Variable | Used by | Purpose |
|---|---|---|
| `THREETEARS_NATS_URL` | host wiring (convention) | NATS URL -- *you* read it and pass to `NatsClient.connect(...)`. |
| `THREETEARS_NATS_PROXY_TIMEOUT_MS` | `NatsProxyL3Backend` | Per-query timeout for proxied L3 (default `5000`). |
| `THREETEARS_LOG_LEVEL`, `THREETEARS_LOG_COLOR` | `threetears.observe` | Standalone logging helpers. |
| `THREETEARS_MCP_TIMEOUT` | `3tears-agent-tools`, `3tears-registry` | MCP tool call timeout. |
| `THREETEARS_REGISTRY_*` | `3tears-registry` | `CALL_TIMEOUT`, `HEALTH_PORT`, `HEARTBEAT_CHECK_INTERVAL`, `HEARTBEAT_TIMEOUT`, `PROBE_TIMEOUT`. |
| `THREETEARS_TOOLSERVER_*`, `THREETEARS_TOOL_SERVER_HEALTH_PORT` | `3tears-agent-tools` | Tool-server readiness/health tuning. |

---

## 11. Observability hookup

3tears follows the standard-library convention: every module logs through a
`threetears` logger with a `NullHandler`, so it is **silent until the host opts in**.

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
tt = logging.getLogger("threetears")
tt.addHandler(handler)
tt.setLevel(logging.INFO)
```

Correlation context and a standalone configurator live in `threetears.observe`:

```python
from threetears.observe import set_context, clear_context, configure_logging
set_context(correlation_id="req-abc", session_id="sess-123", conversation_id="conv-456")
# configure_logging("DEBUG")  # for simple scripts/standalone apps
```

**Tracing** uses a `@traced` decorator on significant entry points. With
OpenTelemetry **not installed** it is a near-zero-overhead passthrough; install OTel
and configure a `TracerProvider` in the host and spans appear automatically.
`BaseCollection` operations set `cache.table` / `cache.hit_tier` attributes.

See [`observability.md`](./observability.md) for the full surface.

---

## 12. Deployment considerations

3tears ships no manifests, charts, or compose files. "Deploying" means standing up
the backing services and injecting connection details. The wiring is identical across
environments; only *what you inject* changes.

### Local development

- **PostgreSQL**: a local instance (Docker is fine). Enable pgvector
  (`CREATE EXTENSION vector;`) only if you use `agent-memory`. For tests, reuse the
  shipped `threetears.core.testing.fixtures` (§8.1).
- **NATS**: **skip it.** Run L1 + L3 (the §8.1 wiring); the cache degrades to L1+L3.
- Inject the L3 DSN however your app already does config.

### Deploying to multiple pods (orchestrator-agnostic)

3tears ships no manifests. Deployment is a sequence of steps; each needs a few inputs
and a matching app-side call. The steps are the same on any orchestrator (Kubernetes,
Nomad, ECS, plain VMs) -- only the mechanism for "run a service", "store a secret", and
"run a one-shot job" differs, which is your platform's concern, not 3tears'.

**Inputs to have on hand:** Postgres DSN · target schema name · (multi-pod) NATS URL,
a subject namespace, a per-pod client name · replica count · (sandboxed pods) the agent
id.

**Step 1 -- Provision PostgreSQL (L3). Required.**
- Inputs: a connection DSN (`postgresql://user:pass@host:5432/db`); the schema your app
  uses; enable the `vector` extension (`CREATE EXTENSION vector`) **only if** you use
  agent-memory.
- App counterpart: `pool = await asyncpg.create_pool(dsn=DSN)` →
  `registry.configure(l3_pool=pool)`.

**Step 2 -- Apply migrations once per schema. Required.**
- Inputs: a single migration runner with your packages registered. Run it as **one**
  one-shot task per schema (not from every pod) so concurrent pods don't race the same
  DDL. It is idempotent and `(version, package)`-keyed, so re-runs are safe.
- App counterpart: `await store.run_migrations(runner)` (=
  `runner.apply_for_agent_schema(store)`, which pins one pooled connection for the run).
  Runs against one database take turns on the database-wide DDL lock, so pods starting
  together are safe; they are serialised, not parallel.

**Step 3 -- Provision NATS (L2). Only for more than one pod, the L3 proxy, or a git L3.**
- Inputs: a NATS URL (`nats://host:4222`); **JetStream enabled**; **persistent (file)
  storage** so the KV bucket survives restarts.
- **One process must DECLARE `{namespace}-collections`; every other process BINDS it.**
  Pick the declaring identity deliberately -- it is whichever process starts first and
  owns the bucket's canonical configuration (a central service, in its own startup). It declares with `ensure_kv_bucket(...)`. A NATS restart DELETES a bucket on
  memory storage, and on Kubernetes a restart that loses the JetStream volume deletes one
  on file storage too; the declaring `NatsClient` creates it again, empty, with the config
  it was declared with, after every reconnect, whatever its storage (it does the same for
  every stream declared through `ensure_jetstream_stream` and every durable consumer it
  bound on one), so the declarer needs no reconnect hook of its own for this. A stream or
  bucket created any other way -- a raw `jetstream_context()` call -- is not remembered,
  and its owner must re-declare it from a reconnect hook. Everyone
  else calls `bind_collections_bucket(...)`, which is `create_if_missing=False` and will
  NOT create it -- so if nothing declares it, every pod dies at wiring, by design.
- App counterpart: `nats = await NatsClient.connect(nats_url=URL,
  nats_subject_namespace=NAMESPACE, client_name=NAME)`; `await
  bind_collections_bucket(nats, component=NAME)`; configure the registry with both
  `l2_client=` and `kv_key_scope=`, so collections pick up L2 (§8.2); call
  `await registry.start_invalidation_listener(nats)` on **every** pod (without it, pods
  serve stale L1 after a peer writes).

**Step 4 -- Supply config as environment variables. Required.**
- Variables the app reads: your Postgres DSN variable; `THREETEARS_NATS_URL`; optionally
  `THREETEARS_NATS_PROXY_TIMEOUT_MS` (sandboxed L3) and `THREETEARS_LOG_LEVEL`. Keep
  secrets out of the image.
- App counterpart: read these in startup wiring and construct the pool / client from
  them; never hard-code.

**Step 5 -- Choose the pod count.**
- Input: replica count. **1 → skip Step 3** and use the §8.1 wiring (L1 + L3). **>1 →
  NATS (Step 3) is required** for cache coherence.

**Step 6 -- Wire startup and shutdown. Required.**
- Inputs: a shutdown grace window **≥ the NATS drain timeout (~30s default)**.
- App counterpart: on startup create the pool and (if multi-pod) the `NatsClient`, then
  `await bind_collections_bucket(nats, component="<your-pod>")` BEFORE
  `registry.configure(...)`, and pass `kv_key_scope=` alongside any `l2_client=` --
  `configure` refuses an L2 client without one, so a process cannot write keys no grant
  names. On
  SIGTERM call `await registry.stop_invalidation_listener()`, then
  `await nats.shutdown()`, then `await pool.close()`. A health check can call
  `await nats.ping()` and run `SELECT 1` on the pool.

**Variant -- sandboxed pods (no DB credentials).** Don't give these pods the DSN. Give
them the NATS URL and the agent id, and configure
`registry.configure(l3_pool=NatsProxyL3Backend(nats_client=nats, namespace_prefix=NAMESPACE, agent_id=AGENT_ID))`.
A separate L3 broker that you provide (§13) holds the real DSN and answers their queries.

**Variant -- cross-pod mutual exclusion.** For singleton jobs / leader election use
`KVLease` (NATS-backed) rather than rolling your own.

### Scaling notes

- L1 is per-pod and in-memory -- scales with pod count for free, no coordination.
- L2 (NATS) is the shared layer that makes horizontal scaling cache-coherent.
- L3 is your durability and throughput ceiling -- size and tune it like any primary DB.

---

## 13. Known gaps

1. **The L3 proxy broker is not part of `threetears.core`.** `NatsProxyL3Backend` is
   only the client; the service answering every `*.l3.*` subject in §3's table must be
   provided separately. Budget for it if you choose the sandboxed-pod topology.

2. **Schema namespacing.** The migration runner and L3 layer set `search_path` to the
   target schema before DDL runs (see `how-to-add-a-migration.md`), so migration/table
   bodies use **unqualified** names. The `DataStore` `agent_id` computes the schema
   name; the pool/broker binds it.

---

## 14. Integration checklist

- [ ] Decide single-pod vs multi-pod (drives whether you deploy NATS).
- [ ] Stand up PostgreSQL; enable pgvector if using `agent-memory`.
- [ ] Create the L3 pool at startup; close it on shutdown.
- [ ] `CollectionRegistry().configure(l1_backend=SQLiteBackend(...), l3_pool=pool)`.
- [ ] (Multi-pod) `await NatsClient.connect(...)`; then
      `await bind_collections_bucket(nats, component="<your-pod>")` BEFORE configuring L2;
      then `registry.configure(..., l2_client=nats, kv_key_scope=kv_key_scope_for(...))`
      so collections pick up L2 (§8.2) -- the scope is REQUIRED wherever an `l2_client` is,
      and `configure` raises `L2ScopeNotConfiguredError` without it;
      `await registry.start_invalidation_listener(nats)` on every pod;
      `await registry.stop_invalidation_listener()` then `await nats.shutdown()` on exit.
- [ ] Define schema with `TableDef`/`ColumnDef`/…; author migrations per
      [`how-to-add-a-migration.md`](./how-to-add-a-migration.md); apply at startup.
- [ ] Choose `collection_flush` (start with `ALWAYS`).
- [ ] Route the `threetears` logger into your logging; optionally configure OTel.
- [ ] Put DSN/NATS URL in secrets; read them in your wiring, not in 3tears.
- [ ] Check §13's known gaps (proxy broker, schema namespacing) against your design.

---

## 15. Appendix: import cheat-sheet

```python
# Core data layer
from threetears.core import (
    CollectionRegistry, DataStore,
    BaseEntity, BaseCollection,
    TableDef, ColumnDef, IndexDef, ForeignKeyDef,
    MigrationRunner,
    CoreConfig, DefaultCoreConfig,
    ConcurrentModificationError, DataLayerUnavailableError,
    create_dynamic_collection,
    KVLease, LeaseHandle,                      # coordination (multi-pod locks)
    Keyset, Page, decode_cursor, encode_cursor, CursorError,   # cursor pagination
    fire_and_forget,                           # submit a coroutine without awaiting it
)

# The eager L2 bucket bind. Every multi-pod process calls this once at startup,
# after connect and BEFORE any configure() that wires an l2_client (see 8.2, 12).
from threetears.core.collections import bind_collections_bucket

# Egress: how outbound traffic leaves. `EgressDriver` is a runtime_checkable Protocol,
# so a host may supply its own exit. Nothing polls `EgressRegistry.health()` for you;
# wire it into your health checks, since a dead exit fails every fetch while looking
# like an application error.
from threetears.core.egress import (
    EgressDriver, EgressRegistry, EgressHealth,
    DirectEgress, ProxyEgress, SocksEgress, WarpEgress,
)

# Migrations (canonical runner)
from threetears.core.data.migrations import (
    MigrationRunner, PackageMigrations, MigrationScope,
)

# Tier backends
from threetears.core.cache.sqlite import SQLiteBackend          # L1 (in-memory)
from threetears.nats import NatsClient                          # L2 (the nats extra)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # L3 (proxy)

# Observability
from threetears.observe import get_logger, traced, set_context, clear_context, configure_logging
```
