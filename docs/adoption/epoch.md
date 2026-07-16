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

- A monotonic, per-subject generation number, durable in Postgres.
- Best-effort broadcast on change.
- The generation number echoed back on every response, so any consumer can
  detect it's stale and lazy-pull the fresh config, even if it missed the
  broadcast.

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
