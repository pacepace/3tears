# 3tears

`threetears.core` -- three-tier caching framework for Python: L1 SQLite, L2
NATS KV, L3 PostgreSQL.

## Problem

Horizontally-scaled apps need every pod to see coherent data fast, without
hand-rolled cache invalidation or per-pod state surgery. Rolling your own
cache layer means re-solving cache coherence, optimistic concurrency, and
schema migrations from scratch, per app, and getting the edge cases wrong
the first few times.

## What it does

- **`BaseEntity`** -- a change-tracking record. Attribute writes mark the
  entity dirty; `save()` persists and promotes through every configured
  tier; optimistic concurrency raises `ConcurrentModificationError` on
  conflict.
- **`BaseCollection`** -- the per-table gateway implementing the three-tier
  read/write logic, with first-class composite primary keys.
- **`DataStore`** -- the ergonomic front door: declares tables, hands out
  collections by name, exposes raw `query`/`execute`, runs migrations.
- **`CollectionRegistry`** -- the one dependency-injection seam. Configure
  L1/L2/L3 backends once; every collection resolves through it, with
  per-table overrides available.
- **Declarative schema** (`TableDef`, `ColumnDef`, `IndexDef`,
  `ForeignKeyDef`) and the canonical `MigrationRunner` -- package-composing,
  topologically ordered, idempotent, version-tracked.
- **Coordination primitives** (`KVLease`) for cross-pod mutual exclusion.

## Design philosophy

A read walks L1 -> L2 -> L3 and re-promotes on the way back. A write goes to
L3 -- the source of truth -- then promotes into L1 and L2, then broadcasts a
cross-pod invalidation. A miss at any tier falls through to the next: the
stack degrades gracefully rather than failing.

**Core makes no connections; the host injects everything it needs to talk
to.** Core never opens a connection or reads an environment variable on your
behalf. You build the backing clients and own their lifecycle; core just
uses them. L1 holds nothing durable -- restart the process and it's empty,
rebuilt from L3. L3 is the only tier you cannot omit.

`3tears-nats` is a hard package dependency of `3tears` -- `pip install
3tears` always installs the NATS client library, whether or not you use it.
What's actually optional is *connecting* to a NATS server: single-pod apps
never call `NatsClient.connect()` or pass `l2_client=` to the registry, so
they run L1 + L3 only. Add a real NATS deployment when you scale to more
than one pod and need cross-pod cache coherence.

Schema is declarative and validated at construction (identifiers must match
`^[a-z][a-z0-9_]*$`; a fixed allowed column-type set). Migrations follow one
blessed path: hand-written, idempotent, version-tracked -- no alembic, no
autogen, no back-compat shims.

## When to adopt

Any app that needs to scale horizontally and wants coherent state across
pods without building its own cache layer. Also useful single-pod, as a
structured alternative to hand-rolled SQLite/Postgres access with
change-tracked entities. Multi-pod deployments need `3tears-nats` for L2.

## Composes with

- [`nats`](nats.md) -- ships as a dependency of `core` either way; a real
  NATS deployment is only required once you connect it as the L2 client for
  cross-pod cache coherence.
- [`observe`](observe.md) -- structured logging and tracing for core's
  operations.
- Nearly every other package in the platform depends on `core` directly or
  indirectly. It is the foundation, not a consumer.

Full wiring guide, decision tables, and copy-pasteable examples:
[`docs/integration-guide.md`](../integration-guide.md).

## Install

```bash
pip install 3tears asyncpg
```
