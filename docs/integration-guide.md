# 3tears Integration Guide

> **Status: DRAFT — re-authored against `develop`.**
>
> Every concrete API, import path, and code sample below was verified against the
> `develop` source tree (the merge target). `develop` moves fast, so **treat code as
> the source of truth** and re-verify before relying on a specific signature. Open
> questions and known sharp edges are in
> [§13](#13-known-sharp-edges--open-questions). Intended for expert review.

This guide explains how to wire 3tears into a host application as its data layer:
the mental model, the design decisions you have to make, and copy-pasteable
"hello world" steps for the L1 / L2 / L3 tiers — for local development and for an
orchestrated (multi-pod) deployment.

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

The monorepo publishes **many** independently-versioned packages (the full table is
in the repo `README.md`). The ones that matter for the data layer:

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

Most wiring funnels through one dependency-injection seam — the
**`CollectionRegistry`** — with one deliberate exception: the **L2 (NATS) client is
passed to each collection at construction**, not through the registry (see §4 and
§13).

```text
        ┌─────────────────────── your host app process ───────────────────────┐
        │                                                                      │
        │   you create:   SQLiteBackend       asyncpg pool      NatsClient*    │
        │                      │                   │                 │         │
        │                      ▼                   ▼                 │         │
        │   CollectionRegistry.configure(l1_backend=, l3_pool=)      │         │
        │                      │                              (constructor arg)│
        │                      ▼                                     ▼         │
        │   DataStore(namespace_id, registry) ─► collections ─► entities       │
        │                                                                      │
        └──────────────────────────────────────────────────────────────────────┘
                * NATS / L2 is optional — single-pod apps skip it (§3, §8.2, §13).
```

---

## 3. The three tiers at a glance

A read walks L1 → L2 → L3 and **re-promotes** on the way back. A write goes to L3
(source of truth), promotes into L1 and L2, and broadcasts a cross-pod invalidation.
A miss at any tier falls through to the next — **the stack degrades gracefully**.

| Tier | Implementation | Nature | Source of truth? | Required? |
|---|---|---|---|---|
| **L1** | `SQLiteBackend` (`threetears.core.cache.sqlite`) | **In-process, in-memory** SQLite (named `memdb` VFS). Fast local cache, **not durable** — rebuilt from L3 on restart. | No | Practically yes (collections cache through it) |
| **L2** | `NatsClient` JetStream KV (`threetears.nats`) | **Cross-pod** shared cache + typed cache-coherence pub/sub. Bucket is `{namespace}-collections`, **file** storage by default (survives broker restarts). | No | **No** — only matters with >1 pod |
| **L3** | an asyncpg-style pool, or `NatsProxyL3Backend` | **Durable storage. The source of truth.** | **Yes** | **Yes, always** |

Consequences:

- **L1 holds nothing durable.** Restart the process and L1 is empty; never treat it
  as a store.
- **Single pod? Skip L2.** With one process there is no cross-pod cache to keep
  coherent. Run L1 + L3 and add NATS when you scale out.
- **L3 is the only thing you cannot omit.**

### Two ways to reach L3

1. **Direct pool** — the process holds DB credentials and you pass a connection pool.
   The pool is **duck-typed**: any object with `async` `.execute()` / `.fetch()` /
   `.fetchrow()` works; `asyncpg` pools satisfy this directly.
2. **Proxy (`NatsProxyL3Backend`)** — the process has **no DB credentials**. It
   serializes SQL and sends it over NATS request/reply (`<ns>.l3.query`,
   `<ns>.l3.batch`) to a central broker that owns the real pool. Use for sandboxed
   worker pods. **Note:** only the *client* ships in `threetears.core`; the broker
   that answers those subjects is **not** part of the package (see §13).

---

## 4. The building blocks

Unless noted, these are exported from `threetears.core` (see §15).

- **`CollectionRegistry`** — DI container + table-name lookup + cache coherence.
  Holds default L1/L3 backends; `configure(l1_backend=, l2_client=, l3_pool=)` sets
  defaults, `bind_table(table_name, ...)` pins per-table overrides **before** a
  collection is constructed (needed because a collection snaps its L3 pool from the
  registry at `__init__`). Drives cross-pod invalidation via
  `start_invalidation_listener(nats)` / `publish_invalidation(...)` using a typed
  `CacheInvalidationMessage` on `Subjects.cache_invalidate()`.
- **`DataStore`** — the ergonomic front door. Wraps a registry, creates tables from
  declarative definitions, hands you collections by name (`store["my_table"]`),
  exposes raw `query` / `execute`, and `run_migrations(runner)`.
- **`BaseEntity`** — a change-tracking record (`entity.field = x` marks it dirty)
  with optimistic concurrency (`date_updated` mismatch → `ConcurrentModificationError`).
  `await entity.save()` / `reload()`; `.id`, `.is_new`.
- **`BaseCollection`** — the per-table gateway implementing the three-tier logic.
  **Composite primary keys are first-class** (`primary_key_column = ("a", "b")`).
  Its `l3_pool` attribute is public — drop to `await self.l3_pool.fetch(...)` for
  ad-hoc SQL the collection API can't express. You rarely subclass it by hand —
  `DataStore` generates one per table.
- **Schema definitions** — `TableDef`, `ColumnDef`, `IndexDef`, `ForeignKeyDef`
  (validated Pydantic models).
- **Migrations** (`threetears.core.data.migrations`) — the canonical
  `MigrationRunner`, `PackageMigrations`, `MigrationScope` (see §9).
- **`CoreConfig` / `DefaultCoreConfig`** — flush-strategy and caching config (a
  `Protocol`, so your own settings object can satisfy it without inheritance).
- **Coordination** — `KVLease` + `LeaseHandle` (NATS-backed distributed lock) for
  multi-pod mutual exclusion.

---

## 5. Design principles

1. **Library instruments, host configures.** You own connections, lifecycle,
   secrets, logging destinations, and trace exporters.
2. **L3 is the source of truth; L1/L2 are disposable.** Losing all caches is a
   performance event, not a correctness event.
3. **Graceful degradation.** A missing/unhealthy cache tier falls through to the
   next. The L2 path narrows its exception scope to real transport errors (`KvError`)
   and degrades to L1+L3; programming errors still surface.
4. **Explicit dependency injection.** One registry, configured once. (L2 is the one
   per-collection constructor arg — see §13.)
5. **Caller owns concurrency boundaries.** Optimistic locking raises on conflict; the
   host decides retry/merge policy. `KVLease` is available for cross-pod mutual
   exclusion.
6. **Schema is declarative and validated.** Identifiers must match
   `^[a-z][a-z0-9_]*$`; column types are a fixed allowed set (§9).
7. **One blessed migration path.** No alembic, no autogen, no back-compat shims —
   hand-written, idempotent, version-tracked migrations (§9).

---

## 6. Decisions you have to make

| Decision | Options | Guidance |
|---|---|---|
| **Single pod or multiple?** | one / many | One → **skip L2/NATS**. Many → add NATS for L2 and cross-pod invalidation. |
| **How does the process reach L3?** | direct pool / NATS proxy | Trusted service with DB creds → direct pool. Sandboxed worker without creds → proxy (and you must provide the broker — §13). |
| **How are tables defined?** | `DataStore` dynamic / hand-written `BaseCollection` | Start with `DataStore`. Drop to hand-written subclasses for bespoke serialization, composite-PK control, or to attach L2 cleanly (§13). |
| **Which packages?** | `core` (+ `3tears-nats`) / + `agent-memory` | KV/relational caching → `core` (+ `3tears-nats` for L2). Semantic memory/search → add `agent-memory` and **enable pgvector**. |
| **Flush strategy** | `ALWAYS`, `ON_CHECKPOINT`, `ON_SCHEDULE`, `ON_SHUTDOWN` | `ALWAYS` = write-through, simplest to reason about. Buffered strategies trade durability latency for throughput (§10). |

---

## 7. Prerequisites & installation

**Host runtime** — Python (check package metadata for the floor) and an `async`
runtime; the data API is `async`/`await` end to end.

**Backing services**

- **PostgreSQL** — always. For `agent-memory` (vector/semantic search):
  `CREATE EXTENSION vector;` (pgvector).
- **NATS with JetStream** — only if you run >1 pod and want L2.

**Install (into your host app's environment)**

```bash
pip install 3tears asyncpg          # core data layer + an L3 driver
pip install 3tears-nats             # L2 client (only if you use NATS)
# pip install 3tears-agent-memory   # agent memory (requires pgvector)
```

For local integration tests, 3tears ships reusable testcontainer fixtures — add
`testcontainers` and see §8.1 / §12.

> The repo's own `uv sync` / `./scripts/*` are for **developing the framework**, not
> for consuming it. As a host app you depend on the published packages.

---

## 8. Hello world

Three progressively richer wirings. Each is self-contained inside an `async` entry
point. CRUD is identical across all three — only the wiring differs.

### 8.1 Minimal: L1 + L3 (single pod, no NATS)

The smallest *correct* configuration: an in-memory L1 cache in front of a durable
PostgreSQL L3. No NATS.

> ⚠️ **Verified caveat (executed against `develop`):** this exact shape — the
> `DataStore` dynamic path with a **raw `asyncpg` pool** and a **`uuid` primary key** —
> does **not** round-trip today. Two framework-level issues bite the L3→L1 re-promotion
> (§13 items 7 & 8). Until they're fixed, the working variants are: give the registry an
> L3 pool that yields **dict rows** (a thin wrapper over the asyncpg pool) **and** use an
> **L1-bindable PK type** such as `text`. The sample below is the *intended* shape; treat
> the `uuid` PK and raw pool as aspirational until §13/7–8 are resolved.

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

    # --- 3. Open a DataStore. First arg is a namespacing id (UUID). ---
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
    got.score = 99                      # change-tracked
    await got.save()

    await widgets.invalidate_cache(wid) # drop from L1 (and L2 if configured)
    refreshed = await widgets.get(wid)  # L1 miss -> L3 -> re-promote
    assert refreshed.score == 99

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
# session-scoped fixtures: db_container -> asyncpg URL (postgres:16),
#                          nats_container -> nats:// URI (JetStream on).
# For pgvector: @pytest.mark.parametrize("db_image", ["pgvector/pgvector:pg16"], indirect=True)
```
They gate on Docker and `pytest.skip` cleanly when it's unavailable.

### 8.2 Add L2 (NATS) for multi-pod caching + coherence

When you run more than one pod, add NATS so pods share an L2 cache **and** a write in
one pod evicts the stale copy in others.

> ⚠️ **Wiring caveat (see §13):** a collection reads its L2 client from the
> **collection constructor**, *not* from `registry.configure(l2_client=...)`. The
> `DataStore.create_table` convenience path does **not** thread a NATS client into
> the collections it builds (those collections log a one-shot warning on first write
> and run L1+L3 only). To attach L2 today, build the dynamic collection yourself with
> `create_dynamic_collection(..., nats_client=...)`. The registry-level invalidation
> listener is independent and works regardless.

```python
import os

from threetears.core import CollectionRegistry, DefaultCoreConfig, TableDef, ColumnDef
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.nats import NatsClient


async def wire_with_l2(pg_pool) -> None:
    l1 = SQLiteBackend(db_name="hello_world")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pg_pool)

    # --- Connect NATS (classmethod). The namespace prefixes the KV bucket,
    #     which collections create lazily as "{namespace}-collections". ---
    nats = await NatsClient.connect(
        nats_url=os.environ["THREETEARS_NATS_URL"],   # e.g. nats://nats:4222
        nats_subject_namespace="myapp",
        client_name="myapp-pod-1",
    )

    config = DefaultCoreConfig(collection_flush="ALWAYS")
    table = TableDef(
        name="widgets",
        columns=[
            ColumnDef(name="id", column_type="uuid", primary_key=True),
            ColumnDef(name="name", column_type="text", nullable=False),
        ],
    )

    # Create the L3 DDL however you prefer (DataStore.create_table, a migration, or
    # raw SQL), then build the collection WITH the NATS client so L2 is active:
    widgets = create_dynamic_collection(
        table_def=table, registry=registry, config=config, nats_client=nats,
    )

    # --- Cross-pod cache coherence: each pod subscribes once at startup so writes
    #     elsewhere evict this pod's stale L1 entry. ---
    await registry.start_invalidation_listener(nats)

    # ... use `widgets` exactly as in §8.1; reads/writes now traverse L1 -> L2 -> L3 ...

    await nats.shutdown()   # graceful drain on shutdown
```

### 8.3 L3 over a proxy (no DB credentials in the pod)

For sandboxed pods that must not hold DB credentials, swap the direct pool for the
NATS proxy backend. Collection code is unchanged — only what you pass as `l3_pool`
differs.

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

Validated Pydantic models:

- **`ColumnDef`**: `name`, `column_type`, `nullable=True`, `default=None`,
  `primary_key=False`. Allowed `column_type`: `text`, `integer`, `bigint`,
  `boolean`, `timestamp`, `uuid`, `jsonb`, `decimal`, `bytea`. *(There is no
  `vector` type — pgvector columns must be added via raw DDL, e.g. `store.execute`,
  not `ColumnDef`.)*
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
# search_path, so migration bodies use unqualified table names.
applied = await runner.apply_for_agent_schema(store)   # or: await store.run_migrations(runner)
# isolation for package-local tests: await runner.apply_package(store, "myapp")
```

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
behavior is set in code. Observed on `develop`:

| Variable | Used by | Purpose |
|---|---|---|
| `THREETEARS_NATS_URL` | host wiring (convention) | NATS URL — *you* read it and pass to `NatsClient.connect(...)`. |
| `THREETEARS_NATS_PROXY_TIMEOUT_MS` | `NatsProxyL3Backend` | Per-query timeout for proxied L3 (default `5000`). |
| `THREETEARS_LOG_LEVEL`, `THREETEARS_LOG_COLOR` | `threetears.observe` | Standalone logging helpers. |
| `THREETEARS_MCP_TIMEOUT` | `3tears-mcp` | MCP tool call timeout. |
| `THREETEARS_REGISTRY_*` | `3tears-registry` | `ACL_TTL_SECONDS`, `CALL_TIMEOUT`, `HEALTH_PORT`, `HEARTBEAT_CHECK_INTERVAL`, `HEARTBEAT_TIMEOUT`, `PROBE_TIMEOUT`. |
| `THREETEARS_TOOLSERVER_*`, `THREETEARS_TOOL_SERVER_HEALTH_PORT` | `3tears-agent-tools` | Tool-server readiness/health tuning. |

---

## 11. Observability hookup

3tears follows the standard-library convention: every module logs through a
`threetears` logger with a `NullHandler`, so it is **silent until the host opts in**.

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(your_formatter)
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
Nomad, ECS, plain VMs) — only the mechanism for "run a service", "store a secret", and
"run a one-shot job" differs, which is your platform's concern, not 3tears'.

**Inputs to have on hand:** Postgres DSN · target schema name · (multi-pod) NATS URL,
a subject namespace, a per-pod client name · replica count · (sandboxed pods) a per-pod
id.

**Step 1 — Provision PostgreSQL (L3). Required.**
- Inputs: a connection DSN (`postgresql://user:pass@host:5432/db`); the schema your app
  uses; enable the `vector` extension (`CREATE EXTENSION vector`) **only if** you use
  agent-memory.
- App counterpart: `pool = await asyncpg.create_pool(dsn=DSN)` →
  `registry.configure(l3_pool=pool)`.

**Step 2 — Apply migrations once per schema. Required.**
- Inputs: a single migration runner with your packages registered. Run it as **one**
  one-shot task per schema (not from every pod) so concurrent pods don't race the same
  DDL. It is idempotent and `(version, package)`-keyed, so re-runs are safe.
- App counterpart: `await store.run_migrations(runner)` (=
  `runner.apply_for_agent_schema(store)`).

**Step 3 — Provision NATS (L2). Only if you run more than one pod.**
- Inputs: a NATS URL (`nats://host:4222`); **JetStream enabled**; **persistent (file)
  storage** so the KV bucket survives restarts. You do **not** create the bucket — the
  app creates `{namespace}-collections` on first use.
- App counterpart: `nats = await NatsClient.connect(nats_url=URL,
  nats_subject_namespace=NAMESPACE, client_name=NAME)`; build collections with
  `nats_client=nats`; call `await registry.start_invalidation_listener(nats)` on **every**
  pod (without it, pods serve stale L1 after a peer writes).

**Step 4 — Supply config as environment variables. Required.**
- Variables the app reads: your Postgres DSN variable; `THREETEARS_NATS_URL`; optionally
  `THREETEARS_NATS_PROXY_TIMEOUT_MS` (sandboxed L3) and `THREETEARS_LOG_LEVEL`. Keep
  secrets out of the image.
- App counterpart: read these in startup wiring and construct the pool / client from
  them; never hard-code.

**Step 5 — Choose the pod count.**
- Input: replica count. **1 → skip Step 3** and use the §8.1 wiring (L1 + L3). **>1 →
  NATS (Step 3) is required** for cache coherence.

**Step 6 — Wire startup and shutdown. Required.**
- Inputs: a shutdown grace window **≥ the NATS drain timeout (~30s default)**.
- App counterpart: on startup create the pool and (if multi-pod) the `NatsClient`; on
  SIGTERM call `await nats.shutdown()` then `await pool.close()`. A health check can call
  `await nats.ping()` and run `SELECT 1` on the pool.

**Variant — sandboxed pods (no DB credentials).** Don't give these pods the DSN. Give
them the NATS URL and a per-pod id, and configure
`registry.configure(l3_pool=NatsProxyL3Backend(nats_client=nats, namespace_prefix=NAMESPACE, agent_id=POD_ID))`.
A separate L3 broker that you provide (§13) holds the real DSN and answers their queries.

**Variant — cross-pod mutual exclusion.** For singleton jobs / leader election use
`KVLease` (NATS-backed) rather than rolling your own.

### Scaling notes

- L1 is per-pod and in-memory — scales with pod count for free, no coordination.
- L2 (NATS) is the shared layer that makes horizontal scaling cache-coherent.
- L3 is your durability and throughput ceiling — size and tune it like any primary DB.

---

## 13. Known sharp edges & open questions

Flagged honestly for expert review. Verify each against current source.

1. **L2 is wired via the collection constructor, not the registry — by design.**
   `BaseCollection` resolves L1/L3 from the registry but reads its NATS client only
   from its constructor. `registry.configure(l2_client=...)` / `get_l2_client()` exist
   and are tested, but **no read/write path in `BaseCollection` consumes them** — the
   collection's L2 comes from `nats_client=`, and the invalidation listener takes the
   client explicitly. A missing client triggers a deliberate one-shot warning
   (DS-06-04), so this is intentional, not silent breakage. *Open question:* is the
   registry's `l2_client` slot vestigial, or intended to become the wiring path?

2. **`DataStore.create_table` does not thread NATS into its collections.** It calls
   `create_dynamic_collection(...)` without a `nats_client`, so collections created
   through `DataStore` have **L2 disabled** (and warn once on first write). To get L2,
   call `create_dynamic_collection(..., nats_client=...)` yourself (§8.2). *Open
   question:* should `DataStore` accept/propagate a NATS client?

3. **L1 `initialize()` is one-shot.** `SQLiteBackend.initialize()` returns early once
   initialized, and `create_dynamic_collection` calls it with a single table's
   metadata. On a **single shared L1 backend**, only the **first** table created
   registers its L1 schema; later tables may be skipped. *Mitigation / open
   question:* use per-table L1 backends via `registry.bind_table(table, l1_backend=...)`,
   or one backend per table — confirm the intended multi-table pattern with
   maintainers. (Single-table examples here avoid the issue.)

4. **The L3 proxy broker is not part of `threetears.core`.** `NatsProxyL3Backend` is
   only the client; the service answering `*.l3.query` / `*.l3.batch` must be provided
   separately. Budget for it if you choose the sandboxed-pod topology.

5. **No `vector` column type in `TableDef`.** pgvector columns (for `agent-memory`)
   must be created via raw DDL, not `ColumnDef`. Confirm the supported path for vector
   schema if you build on agent-memory.

6. **Schema namespacing — resolved.** Earlier drafts asked how multi-tenant schema
   isolation works. The migration runner and L3 layer set `search_path` to the target
   schema before DDL runs (see `how-to-add-a-migration.md`), so migration/table bodies
   use **unqualified** names. The `DataStore` `agent_id` computes the schema name; the
   pool/broker binds it.

7. **CONFIRMED BUG (executed) — dynamic collections return raw asyncpg `Record`s, which
   break L1 re-promotion.** `create_dynamic_collection.fetch_from_postgres` returns
   `rows[0]` (an `asyncpg.Record`) unchanged. On an L1 miss, `_pull_through` passes it to
   `SQLiteBackend.upsert`, which does `[c for c in data if c in schema]` — but iterating
   an `asyncpg.Record` yields **values, not keys**, so it builds
   `INSERT INTO <t> () VALUES ()` → `sqlite3.OperationalError: near ")"`. The repo's own
   integration test masks this by using a **dict-returning stub pool**. Likely fix:
   `fetch_from_postgres` should return `dict(rows[0])`. *Workaround:* wrap the L3 pool so
   `fetch`/`fetchrow` return `dict(row)`.

8. **CONFIRMED BUG (executed) — asyncpg `pgproto.UUID` PK values can't bind to L1.** Even
   with dict rows, asyncpg deserializes `uuid` columns to `asyncpg.pgproto.pgproto.UUID`.
   `SQLiteBackend.select_by_id` binds the PK value **raw** (no `serialize_value` on the
   way in), and sqlite3 only has an adapter for stdlib `uuid.UUID`, so a re-promoted
   uuid id raises `sqlite3.ProgrammingError: type 'asyncpg.pgproto.pgproto.UUID' is not
   supported`. *Workaround:* use a `text` PK, or register an sqlite3 adapter for
   asyncpg's UUID. Likely fix: serialize PK values at the L1 boundary the same way
   `upsert` serializes column values. (Items 7 + 8 verified end-to-end against a
   testcontainers Postgres: raw-pool+uuid fails, dict-pool+uuid fails, dict-pool+text
   round-trips.)

---

## 14. Integration checklist

- [ ] Decide single-pod vs multi-pod (drives whether you deploy NATS).
- [ ] Stand up PostgreSQL; enable pgvector if using `agent-memory`.
- [ ] Create the L3 pool at startup; close it on shutdown.
- [ ] `CollectionRegistry().configure(l1_backend=SQLiteBackend(...), l3_pool=pool)`.
- [ ] (Multi-pod) `await NatsClient.connect(...)`; build collections with
      `nats_client=` (§8.2 caveat); `await registry.start_invalidation_listener(nats)`
      on every pod; `await nats.shutdown()` on exit.
- [ ] Define schema with `TableDef`/`ColumnDef`/…; author migrations per
      [`how-to-add-a-migration.md`](./how-to-add-a-migration.md); apply at startup.
- [ ] Choose `collection_flush` (start with `ALWAYS`).
- [ ] Route the `threetears` logger into your logging; optionally configure OTel.
- [ ] Put DSN/NATS URL in secrets; read them in your wiring, not in 3tears.
- [ ] Re-read §13 and verify the sharp edges that touch your design.

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
)

# Migrations (canonical runner)
from threetears.core.data.migrations import (
    MigrationRunner, PackageMigrations, MigrationScope,
)

# Tier backends
from threetears.core.cache.sqlite import SQLiteBackend          # L1 (in-memory)
from threetears.nats import NatsClient                          # L2 (3tears-nats)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # L3 (proxy)

# Observability
from threetears.observe import get_logger, traced, set_context, clear_context, configure_logging
```
