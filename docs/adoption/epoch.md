# 3tears-epoch

`threetears.epoch` -- generation-stamped config epochs with NATS broadcast
and per-message echo, for coherent cross-pod cache reloads.

## Problem

In-memory config caches (a model registry, a catalog cache, MCP RBAC grants)
need every pod to reload when the config changes. Pure push (broadcast a
NATS message) has a missed-message hole -- a pod that's briefly
disconnected never reloads. Pure pull (poll on an interval) is correct but
expensive at scale.

## What it does

- A monotonic, per-subject generation number, counted in a memory-backed
  NATS KV bucket.
- Best-effort broadcast on change.
- The generation number echoed back on every response, so any consumer can
  detect it's stale and lazy-pull the fresh config, even if it missed the
  broadcast.

## Epochs are not durable

An epoch resets when the broker restarts, and that is the design: an epoch is
a coherence signal, and a restart that loses the counter also loses every
cache it was sequencing. Nothing raises when it happens. The bucket is
recreated empty, every KV call keeps succeeding, and the counter reads zero.

The counter therefore carries an opaque identity key. `EpochListener`
compares it for equality and, when it changes, clears last-seen and fires
`on_reset` for every registration. Wire that callback:

```python
await listener.subscribe(subject, on_bump, on_reset=my_reload_callback)
```

`on_reset` takes no epoch and must not be epoch-deduped. A reset means the
counter you were comparing against no longer exists, so the only number
available is below everything you have already acted on, which is exactly
what dedupe discards.

Detection needs a poll to run in. `3tears-epoch` cannot own a task inside
`3tears`, so the loop is yours and the pass is ours:

```python
from threetears.epoch import catchup_tick

# in your own loop, at your own interval
await catchup_tick(listener, [(subject, on_bump), ...])
```

A consumer that subscribes and schedules nothing still receives broadcasts,
but misses everything a broadcast can lose, including a replaced counter.

### There is no durable epoch in NATS, deliberately

`NatsKvBucket` accepts `storage="file"`, so a counter that survives a broker
process restart is constructible. It is not used, because file-backed JetStream
survives only if the store directory survives, and the failure this design
answers is a broker on ephemeral storage whose restart wipes JetStream
wholesale. On that deployment `file` is exactly as durable as `memory`, so NATS
durability is conditional on how someone provisioned a volume.

For every epoch except one, that does not matter: a reset costs one extra
reload from a lower tier that is the real source of truth, and fails safe. For
the one whose value escapes to caches we cannot purge, conditional durability is
the wrong guarantee, so it keeps an unconditional Postgres row.

Which substrate a subject takes is declared per family in
`threetears.epoch.client`, not passed per call: a per-call flag lets two call
sites disagree about one subject. The classifier still matches a path marker to
apply that declaration, so it is the enumeration test named below, not the
matcher, that stops a subject nobody considered from being classified silently.
`packages/epoch/tests/unit/test_durability_policy.py` enumerates the real
`Subjects` factory and fails when a new `*_epoch` builder is in neither table,
so adding one forces the decision rather than defaulting to ephemeral -- the
direction that cannot be repaired.

One family is carved out and stays on a durable Postgres row:
`Subjects.datasource_tile_epoch`. Its value is the `v{n}` in a tile URL and
reaches CDN and browser caches, so a counter that reset would re-issue
`v1..vN` for different content while those caches still hold the old
generation. Routing is by subject shape, because durability is a property of
what the number means rather than of who bumps it.

## Deployment

Every principal that reads or bumps an epoch needs the `{namespace}-epochs`
KV bucket granted: `AGENT_POD`, `HUB`, `GATEWAY`. It is already in
`threetears.nats.subject_permissions`, so minting JWTs from that module
covers it. If your deployment declares bucket grants anywhere else, add it
there too.

Miss it and the epoch calls time out while everything else on the same
connection stays healthy. The wrapper names the cause in the log rather than
leaving you to infer it; see [`nats`](nats.md), "KV bucket grants".

## Design philosophy

Combines both push and pull to get the correctness of pull with the latency
of push, rather than picking one and accepting its failure mode. This is
explicitly modeled on prior art that solves the same problem elsewhere:
etcd's `mod_revision`, Kubernetes's `resourceVersion` / informer pattern,
Envoy's xDS, and DNS SOA serial numbers. The pattern generalizes to any
in-memory cache that must stay coherent across pods without a full poll
loop.

## When to adopt

Any multi-pod deployment with an in-memory config cache that must reload
promptly and correctly when the underlying config changes -- model
registries, RBAC grant caches, catalogs.

## Composes with

- [`nats`](nats.md) -- the broadcast transport.
- [`mcp`](mcp.md) -- uses epoch broadcast for tool-grant cache reload.

## Install

```bash
pip install 3tears-epoch
```
