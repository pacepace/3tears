# 3tears Integration Guide

> **Status: DRAFT.** This is an early, source-derived starting point intended for
> expert review. It documents the framework as the code currently behaves, including
> several sharp edges flagged in [§13 Known sharp edges & open questions](#13-known-sharp-edges--open-questions).
> Treat code as the source of truth; where this guide and the code disagree, the code wins —
> and please file a correction.

This guide explains how to wire 3tears into a host application as its data layer:
the mental model, the design decisions you have to make, and copy-pasteable
"hello world" steps for the L1 / L2 / L3 tiers — both for local development and
for an orchestrated (multi-pod) deployment.

It is deliberately generic. It assumes no particular host app, cloud, or platform.

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
13. [Known sharp edges & open questions](#13-known-sharp-edges--open-questions)
14. [Integration checklist](#14-integration-checklist)
15. [Appendix: import cheat-sheet](#15-appendix-import-cheat-sheet)

---

## 1. What 3tears is (and is not)

3tears is a **Python library suite**, not a service.

- There is **no daemon, container, sidecar, or API** to deploy for 3tears itself.
- You `pip install` the packages you need and construct objects **inside your host
  application's process**.
- It brings **infrastructure dependencies** (PostgreSQL, optionally a NATS server),
  but *those* are services you stand up and own — 3tears just talks to them.

The packages (install only what you use):

| Package (PyPI) | Import root | What it gives you |
|---|---|---|
| `3tears` | `threetears.core` | Three-tier entities/collections, caching (L1/L2/L3), `DataStore`, schema & migrations |
| `3tears-observe` | `threetears.observe` | Structured logging, OpenTelemetry `@traced` |
| `3tears-agent-memory` | `threetears.agent.memory` | LLM agent memory: extraction, retrieval, hybrid search (needs pgvector) |
| `3tears-agent-tools` | `threetears.agent.tools` | Tool ABC, NATS tool server, built-in tools |
| `3tears-langgraph` | `threetears.langgraph` | LangGraph checkpoint savers & graph builders |
| `3tears-registry` | `threetears.registry` | Multi-pod tool catalog, discovery, load balancing (needs NATS) |

This guide focuses on **`threetears.core`** — the data layer. The other packages
build on the same wiring described here.

---

## 2. The core mental model

> **The library instruments; the host configures.**

3tears never opens a database connection, reads an environment variable for
config, or installs a log handler on your behalf. **You** construct the backing
clients (a PostgreSQL pool, optionally a NATS client), hand them to 3tears, and
**you own their lifecycle** (create on startup, close on shutdown).

Everything funnels through one dependency-injection seam: the **`CollectionRegistry`**.
You `configure()` it with your tier backends once, and every collection resolves
its dependencies through it.

```text
        ┌─────────────────────── your host app process ───────────────────────┐
        │                                                                      │
        │   you create:        SQLiteBackend   NatsClient*    asyncpg pool     │
        │                            │              │              │           │
        │                            ▼              ▼              ▼           │
        │   CollectionRegistry.configure(l1_backend=, l2_client=*, l3_pool=)   │
        │                            │                                         │
        │                            ▼                                         │
        │   DataStore(namespace_id, registry)  ──►  collections  ──► entities  │
        │                                                                      │
        └──────────────────────────────────────────────────────────────────────┘
                * NATS / L2 is optional — see §3 and §13.
```

---

## 3. The three tiers at a glance

A read walks L1 → L2 → L3 and **re-promotes** on the way back. A write goes to L3
(source of truth) and then promotes into L1 and L2. A cache miss at any tier
simply falls through to the next — **the stack degrades gracefully**.

| Tier | Implementation | Nature | Source of truth? | Required? |
|---|---|---|---|---|
| **L1** | `SQLiteBackend` | **In-process, in-memory** SQLite (named `memdb` VFS). Fast local cache. **Not durable** — it is a cache, not storage. | No | Practically yes (collections cache through it) |
| **L2** | `NatsClient` (NATS JetStream KV) | **Cross-pod** shared cache + cache-coherence pub/sub. Bucket uses **file** storage so it survives NATS restarts (TTL 7200s by default). | No | **No** — only matters with more than one pod |
| **L3** | a PostgreSQL connection pool (e.g. `asyncpg`), or `NatsProxyL3Backend` | **Durable storage. The source of truth.** | **Yes** | **Yes, always** |

Key consequences:

- **L1 holds nothing durable.** Restart the process and L1 is empty; it rebuilds
  from L3. Never treat L1 as a store.
- **Single pod? Skip L2.** With one process there is no cross-pod cache to keep
  coherent. Run L1 + L3 and add NATS later if you scale out.
- **L3 is the only thing you cannot omit.** Everything ultimately persists there.

### Two ways to reach L3

1. **Direct pool** — the process holds DB credentials and you pass a connection
   pool (anything with `async` `.execute()` / `.fetch()` / `.fetchrow()` — `asyncpg`
   fits as-is; the pool is **duck-typed**, not a fixed class). Simple; use for
   trusted services.
2. **Proxy (`NatsProxyL3Backend`)** — the process has **no DB credentials**. It
   serializes SQL and sends it over NATS request/reply (`<ns>.l3.query`,
   `<ns>.l3.batch`) to a central broker that owns the real pool. Use for sandboxed
   worker pods. **Note:** only the *client* side ships in `threetears.core`; the
   broker that answers those subjects is **not** part of this package (see §13).

---

## 4. The building blocks

All of these are exported from `threetears.core` (see §15).

- **`CollectionRegistry`** — the DI container and table-name lookup. Holds default
  L1/L3 backends; supports per-table overrides via `register(...)`. Also drives
  cross-pod cache coherence (`start_invalidation_listener`, `publish_invalidation`).
- **`DataStore`** — the ergonomic front door. Wraps a registry, creates tables
  from declarative definitions, and hands you collections by name
  (`store["my_table"]`). Construct with a namespacing id and the registry.
- **`BaseEntity`** — a smart record with **change tracking** (`entity.field = x`
  marks it dirty) and **optimistic concurrency** (`date_updated` mismatch →
  `ConcurrentModificationError`). `await entity.save()` / `reload()` / `delete()`.
- **`BaseCollection`** — the per-table gateway implementing the three-tier
  read/write/promote/invalidate logic. You rarely subclass it by hand — `DataStore`
  generates one per table for you.
- **Schema definitions** — `TableDef`, `ColumnDef`, `IndexDef`, `ForeignKeyDef`
  (Pydantic models, validated).
- **`MigrationRunner`** — version-tracked, idempotent schema migrations.
- **`CoreConfig` / `DefaultCoreConfig`** — flush-strategy and caching config
  (a `Protocol`, so your own settings object can satisfy it without inheritance).

---

## 5. Design principles

These are the framework's load-bearing assumptions. Designing with them avoids
most integration pain.

1. **Library instruments, host configures.** You own connections, lifecycle,
   secrets, logging destinations, and trace exporters. 3tears assumes nothing about
   your deployment.
2. **L3 is the source of truth; L1/L2 are disposable.** Design so that losing all
   cache tiers is a performance event, not a correctness event.
3. **Graceful degradation.** A missing/unhealthy cache tier falls through to the
   next. The NATS client is **fail-open**: its KV operations return `None`/`False`
   on error rather than raising, so a NATS outage degrades to L1 + L3 instead of
   taking the app down.
4. **Explicit dependency injection.** There is no global singleton and no implicit
   connection discovery. One registry, configured once, threaded everywhere.
5. **Caller owns concurrency boundaries.** Optimistic locking surfaces conflicts as
   exceptions; the host decides retry/merge policy.
6. **Schema is declarative and validated.** Identifiers must match
   `^[a-z][a-z0-9_]*$`; column types are a fixed allowed set (§9). Definitions are
   rejected early rather than failing in SQL.

---

## 6. Decisions you have to make

Make these before writing wiring code:

| Decision | Options | Guidance |
|---|---|---|
| **Single pod or multiple?** | one process / many | One → **skip L2/NATS**. Many → add NATS for L2 and cross-pod invalidation. |
| **How does the process reach L3?** | direct pool / NATS proxy | Trusted service with DB creds → direct pool. Sandboxed worker without creds → proxy (and you must provide the broker — §13). |
| **How are tables defined?** | `DataStore` dynamic collections / hand-written `BaseCollection` subclasses | Start with `DataStore` — it generates collections and initializes L1 from your `TableDef`. Drop to hand-written subclasses only for bespoke serialization or wiring (e.g. to attach L2 today — see §13). |
| **Which packages?** | `core` only / + `agent-memory` / + others | Plain key-value/relational caching → `core`. Semantic memory/search → add `agent-memory` and **enable pgvector** in Postgres. |
| **Flush strategy** | `ALWAYS`, `ON_CHECKPOINT`, `ON_SCHEDULE`, `ON_SHUTDOWN` | `ALWAYS` is simplest to reason about (write-through). Buffered strategies trade durability latency for throughput — see §10. |

---

## 7. Prerequisites & installation

**Host runtime**

- Python (the framework targets a recent 3.x; check the package metadata for the
  exact floor).
- An `async` runtime — the data API is `async`/`await` end to end.

**Backing services**

- **PostgreSQL** — always. For `agent-memory` (vector/semantic search) you must
  `CREATE EXTENSION vector;` (pgvector).
- **NATS with JetStream** — only if you run more than one pod and want L2.

**Install (into your host app's environment)**

```bash
pip install 3tears          # threetears.core
pip install asyncpg         # a PostgreSQL driver for the L3 pool (duck-typed; any equivalent works)
# pip install 3tears-agent-memory   # if you need agent memory (requires pgvector)
```

> The 3tears repo's own `uv sync` / `./scripts/*` are for **developing the
> framework**, not for consuming it. As a host app you depend on the published
> packages.

---

## 8. Hello world

Three progressively richer wirings. Each is self-contained and runnable inside an
`async` entry point.

### 8.1 Minimal: L1 + L3 (single pod, no NATS)

This is the smallest *correct* configuration: an in-memory L1 cache in front of a
durable PostgreSQL L3. No NATS.

```python
import asyncio
import uuid

import asyncpg

from threetears.core import (
    CollectionRegistry,
    DataStore,
    DefaultCoreConfig,
    TableDef,
    ColumnDef,
    IndexDef,
)
from threetears.core.cache.sqlite import SQLiteBackend


async def main() -> None:
    # --- 1. Build the tier backends (you own these) ---
    l1 = SQLiteBackend(db_name="hello_world")               # L1: in-process, in-memory
    pg_pool = await asyncpg.create_pool(                    # L3: durable source of truth
        dsn="postgresql://user:pass@localhost:5432/appdb",
    )

    # --- 2. Configure the one DI seam ---
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pg_pool)     # note: no l2_client here

    # --- 3. Open a DataStore. The first arg is a namespacing id (UUID). ---
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
    await entity.save()          # writes L3, promotes into L1

    got = await widgets.get(wid) # L1 hit
    got.score = 99               # change-tracked
    await got.save()

    await widgets.invalidate_cache(wid)   # drop from L1 (and L2 if configured)
    refreshed = await widgets.get(wid)    # L1 miss → L3 → re-promote
    assert refreshed.score == 99

    await widgets.delete(wid)             # removes from every tier

    await pg_pool.close()        # you own shutdown


asyncio.run(main())
```

What just happened:

- `create_table` built and ran the `CREATE TABLE`/index DDL against L3, then
  initialized L1's in-memory schema from the same `TableDef`.
- `create()` returns an unsaved, dirty entity; `save()` persists it and promotes it
  into the caches; `get()` reads through the tiers and re-promotes on miss.
- With `collection_flush="ALWAYS"`, every `save()` is write-through (no buffering).

### 8.2 Add L2 (NATS) for multi-pod caching + coherence

When you run more than one pod, add NATS so the pods share an L2 cache **and** so a
write in one pod evicts the stale copy in others.

> ⚠️ **Important wiring caveat (verify — see §13):** a collection reads its L2
> client from the **collection constructor**, *not* from `registry.configure(l2_client=...)`.
> The `DataStore.create_table` convenience path does **not** currently thread a NATS
> client into the collections it builds. To attach L2 **today**, construct the
> dynamic collection yourself with `create_dynamic_collection(..., nats_client=...)`,
> or use a hand-written `BaseCollection` subclass. The registry-level **invalidation
> listener** below is independent of that and works regardless.

```python
import os

from threetears.core import CollectionRegistry, DefaultCoreConfig, TableDef, ColumnDef
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.cache.nats import NatsClient


async def wire_with_l2(pg_pool) -> None:
    l1 = SQLiteBackend(db_name="hello_world")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pg_pool)

    # --- Connect NATS. This ensures the JetStream KV bucket exists
    #     (`<prefix>-collections`, file storage, TTL 7200s). ---
    nats = NatsClient(bucket_prefix="myapp")
    await nats.connect(os.environ["THREETEARS_NATS_URL"])   # e.g. nats://nats:4222

    config = DefaultCoreConfig(collection_flush="ALWAYS")
    table = TableDef(
        name="widgets",
        columns=[
            ColumnDef(name="id", column_type="uuid", primary_key=True),
            ColumnDef(name="name", column_type="text", nullable=False),
        ],
    )

    # Create the table's L3 DDL however you prefer (DataStore.create_table, or raw SQL),
    # then build the collection WITH the NATS client so L2 is active:
    widgets = create_dynamic_collection(
        table_def=table, registry=registry, config=config, nats_client=nats,
    )

    # --- Cross-pod cache coherence: each pod subscribes once at startup so writes
    #     elsewhere evict this pod's stale L1 entry. ---
    await registry.start_invalidation_listener(nats)

    # ... use `widgets` exactly as in 8.1; reads/writes now traverse L1 → L2 → L3 ...

    await nats.close()   # drain on shutdown
```

### 8.3 L3 over a proxy (no DB credentials in the pod)

For sandboxed pods that must not hold DB credentials, swap the direct pool for the
NATS proxy backend. The collection code is unchanged — only what you pass as
`l3_pool` differs.

```python
from threetears.core.backends.nats_proxy import NatsProxyL3Backend

l3 = NatsProxyL3Backend(
    nats_client=nats,            # a connected NATS client
    namespace_prefix="myapp",    # subjects become myapp.l3.query / myapp.l3.batch
    agent_id=str(pod_id),        # used for ACL/namespacing at the broker
)
registry.configure(l1_backend=l1, l3_pool=l3)
```

> The broker that *answers* `myapp.l3.query` (owns the real PostgreSQL pool, executes
> SQL, returns rows) is **not** included in `threetears.core`. You must provide it.
> See §13.

---

## 9. Schema & migrations

### Declarative schema

Tables are declared with validated Pydantic models:

- **`ColumnDef`**: `name`, `column_type`, `nullable=True`, `default=None`,
  `primary_key=False`.
  Allowed `column_type` values: `text`, `integer`, `bigint`, `boolean`,
  `timestamp`, `uuid`, `jsonb`, `decimal`, `bytea`.
- **`IndexDef`**: `name`, `columns`, `unique=False`.
- **`ForeignKeyDef`**: `name`, `columns`, `references_table`, `references_columns`,
  `on_delete="CASCADE"`, `on_update="NO ACTION"` (actions: `CASCADE`, `SET NULL`,
  `RESTRICT`, `NO ACTION`).
- **`TableDef`**: `name`, `columns`, `indexes=[]`, `foreign_keys=[]`.

All identifiers must match `^[a-z][a-z0-9_]*$`. Invalid definitions raise at
construction time, not at SQL time.

### Versioned migrations

`MigrationRunner` is **constructed with a `DataStore`** (not a raw pool). Register
migrations with the `@version(n)` decorator; `apply()` creates a `_schema_migrations`
tracking table, runs only the pending versions in ascending order, and is safe to
run on every startup (idempotent).

```python
from threetears.core import DataStore, MigrationRunner, TableDef, ColumnDef

runner = MigrationRunner(store)   # store is a DataStore

@runner.version(1)
async def initial_schema(store: DataStore) -> None:
    await store.create_table(TableDef(
        name="users",
        columns=[
            ColumnDef(name="id", column_type="uuid", primary_key=True),
            ColumnDef(name="name", column_type="text", nullable=False),
        ],
    ))

@runner.version(2)
async def add_email(store: DataStore) -> None:
    await store.execute("ALTER TABLE users ADD COLUMN email TEXT")

applied = await runner.apply()      # or: await store.run_migrations(runner)
# helpers: await runner.current_version(), await runner.pending()
```

---

## 10. Configuration reference

### `CoreConfig` (in-code configuration)

`CoreConfig` is a `runtime_checkable` `Protocol`; `DefaultCoreConfig` is the
concrete default. Any object exposing these three attributes satisfies it.

| Field | Type | Default (`DefaultCoreConfig`) | Meaning |
|---|---|---|---|
| `collection_flush` | `str` | `"ON_CHECKPOINT"` | Write-buffer strategy. One of `ALWAYS`, `ON_CHECKPOINT`, `ON_SCHEDULE`, `ON_SHUTDOWN`. `ALWAYS` = write-through. |
| `collection_flush_interval` | `int` (s) | `30` | Interval for `ON_SCHEDULE` flushing. |
| `collection_flush_tables` | `str` | `"messages,token_usage_logs"` | Comma-separated tables the flush strategy applies to. |

> For predictable durability while learning the framework, use
> `DefaultCoreConfig(collection_flush="ALWAYS")`. Move to buffered strategies once
> you understand the durability/throughput trade-off for a given table.

### Environment variables

3tears reads a small set of env vars **as fallbacks/tuning**; in keeping with
"host configures," most behavior is set in code. Observed variables:

| Variable | Used by | Purpose |
|---|---|---|
| `THREETEARS_NATS_URL` | host wiring (convention) | NATS connection URL. *You* read it and pass it to `NatsClient.connect(...)`. |
| `THREETEARS_NATS_PROXY_TIMEOUT_MS` | `NatsProxyL3Backend` | Per-query timeout for proxied L3 (default `5000`). |
| `THREETEARS_LOG_LEVEL`, `THREETEARS_LOG_COLOR` | `threetears.observe` logging | Standalone logging helpers. |
| `THREETEARS_MCP_TIMEOUT` | `agent-tools` (MCP) | MCP tool call timeout. |
| `THREETEARS_REGISTRY_CALL_TIMEOUT`, `THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL`, `THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT` | `3tears-registry` | Tool-registry call & heartbeat tuning. |

---

## 11. Observability hookup

3tears follows the standard library convention: every module logs through a
`threetears` logger with a `NullHandler`, so it is **silent until the host opts in**.

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(your_formatter)
tt = logging.getLogger("threetears")
tt.addHandler(handler)
tt.setLevel(logging.INFO)
```

For correlation, set the provided context vars at request entry:

```python
from threetears.core.logging import set_context
set_context(correlation_id="req-abc", session_id="sess-123", conversation_id="conv-456")
```

**Tracing** uses a `@traced` decorator on significant entry points. With
OpenTelemetry **not installed** it is a zero-overhead passthrough; install OTel and
configure a `TracerProvider` in the host and spans appear automatically — 3tears
needs no further configuration. Sensitive argument names (password/token/secret/…)
are redacted when argument recording is enabled.

See [`observability.md`](./observability.md) for the full surface (traced methods,
span attributes, standalone helpers).

---

## 12. Deployment considerations

3tears ships no manifests, charts, or compose files. "Deploying" means standing up
the backing services and injecting connection details. The wiring is identical
across environments; only *what you inject* changes.

### Local development

- **PostgreSQL**: a local instance is enough. Enable pgvector
  (`CREATE EXTENSION vector;`) only if you use `agent-memory`.
- **NATS**: **skip it.** Run L1 + L3 (the §8.1 wiring). The cache degrades to
  "L1 + L3" with no code change.
- Point the L3 DSN at your local Postgres; pass connection details however your app
  already does config (env, `.env`, settings object).

### Orchestrated / multi-pod (e.g. Kubernetes)

- **PostgreSQL + pgvector**: a managed service or an in-cluster operator/StatefulSet.
  Inject the DSN via a secret → env var → your L3 pool. **Size for your corpus**;
  vector indexes only start to matter at scale.
- **NATS with JetStream**: deploy once you run **more than one pod**. JetStream must
  be **enabled with file storage** so the `*-collections` KV bucket survives broker
  restarts. Inject the URL (e.g. `THREETEARS_NATS_URL`).
- **Cache coherence is opt-in:** call `registry.start_invalidation_listener(nats)`
  on **every** pod at startup. Without it, pods can serve stale L1 data after a peer
  writes.
- **Choose your L3 topology per pod class:**
  - trusted services → direct pool with DB credentials;
  - sandboxed workers → `NatsProxyL3Backend` + a central broker you provide (§13).
- **Secrets**: PostgreSQL DSN and NATS URL belong in your platform's secret store,
  surfaced as env vars and read by *your* wiring code — never hard-coded into 3tears.
- **Lifecycle**: create the pool / NATS client during app startup; **close/drain
  them on shutdown**. 3tears does not manage process lifecycle.

### Scaling notes

- L1 is per-pod and in-memory — it scales with pod count for free and needs no
  coordination.
- L2 (NATS) is the shared layer; it is the thing that makes horizontal scaling
  cache-coherent.
- L3 is your durability and throughput ceiling — size and tune it like any primary
  database.

---

## 13. Known sharp edges & open questions

Flagged honestly for expert review. Verify each against the current source before
relying on it.

1. **L2 is wired via the collection constructor, not the registry.**
   `BaseCollection` resolves L1 and L3 from the registry but reads its NATS client
   only from its constructor (`nats_client=`). `registry.configure(l2_client=...)`
   and `registry.get_l2_client()` exist and are covered by tests, but **no
   read/write path in `BaseCollection` consumes them** — they appear unused outside
   tests. *Open question:* is `registry.l2_client` intended to be the wiring path
   (and the collection should read it), or is the constructor the intended path?

2. **`DataStore.create_table` does not thread NATS into its collections.** It calls
   `create_dynamic_collection(...)` without a `nats_client`, so collections created
   through `DataStore` have **L2 disabled** even if NATS is connected. To get L2
   today, call `create_dynamic_collection(..., nats_client=...)` yourself or use a
   hand-written collection. *Open question:* should `DataStore` accept/propagate a
   NATS client?

3. **L1 `initialize()` is one-shot.** `SQLiteBackend.initialize()` returns early if
   already initialized, and `create_dynamic_collection` calls it with **one table's**
   metadata. On a **single shared L1 backend**, only the **first** table created may
   register its L1 schema; subsequent tables' L1 schema may be skipped. *Open
   question / verify:* what is the intended pattern for **multiple tables on one L1
   backend** — initialize once with combined metadata, a per-table/per-collection L1
   backend via `registry.register(..., l1_backend=...)` overrides, or something
   else? This guide's single-table examples avoid the issue; multi-table apps should
   confirm the L1 behavior.

4. **The L3 proxy broker is not part of `threetears.core`.** `NatsProxyL3Backend`
   is only the client. The service that answers `*.l3.query` / `*.l3.batch` and owns
   the real PostgreSQL pool must be provided separately. Budget for it if you choose
   the sandboxed-pod topology.

5. **L3 pool is duck-typed.** Any object with `async` `.execute()`, `.fetch()`,
   `.fetchrow()` works; `asyncpg` pools satisfy this. There is no formal adapter
   interface to implement against — match the methods the collections call.

6. **Schema namespacing.** `DataStore` computes a per-namespace schema name from its
   id, but the table DDL paths observed use bare table names. Confirm how
   multi-tenant schema isolation / `search_path` is expected to be set up (likely a
   host responsibility on the pool) before relying on per-namespace isolation.

---

## 14. Integration checklist

- [ ] Decide single-pod vs multi-pod (drives whether you deploy NATS).
- [ ] Stand up PostgreSQL; enable pgvector if using `agent-memory`.
- [ ] Create the L3 connection pool at startup; arrange to close it on shutdown.
- [ ] `CollectionRegistry().configure(l1_backend=SQLiteBackend(...), l3_pool=pool)`.
- [ ] (Multi-pod) Connect `NatsClient`; attach it to collections (§8.2 caveat) and
      call `registry.start_invalidation_listener(nats)` on every pod.
- [ ] Define schema with `TableDef`/`ColumnDef`/…; run `MigrationRunner.apply()` at
      startup.
- [ ] Choose `collection_flush` (start with `ALWAYS`).
- [ ] Route the `threetears` logger into your logging; optionally configure OTel.
- [ ] Put DSN/NATS URL in secrets; read them in your wiring, not in 3tears.
- [ ] Re-read §13 and verify the sharp edges that touch your design.

---

## 15. Appendix: import cheat-sheet

```python
# Core data layer (all from threetears.core)
from threetears.core import (
    CollectionRegistry, DataStore,
    BaseEntity, BaseCollection,
    TableDef, ColumnDef, IndexDef, ForeignKeyDef,
    MigrationRunner,
    CoreConfig, DefaultCoreConfig,
    ConcurrentModificationError, DataLayerUnavailableError,
    create_dynamic_collection,
)

# Tier backends
from threetears.core.cache.sqlite import SQLiteBackend          # L1
from threetears.core.cache.nats import NatsClient, BucketConfig  # L2
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # L3 (proxy)

# Optional models / mixins
from threetears.core.models import UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin

# Observability
from threetears.core.logging import set_context, configure_logging
```
