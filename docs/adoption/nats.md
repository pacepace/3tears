# 3tears-nats

`threetears.nats` -- typed NATS client, subject builders, message envelopes,
and JetStream KV bucket helpers.

## Problem

NATS's raw Python client is easy to misuse in ways that fail silently. A
callback can be mistaken for a queue group. Raw strings for subjects invite
typos that never surface until a message goes nowhere. Every app that
touches NATS ends up re-solving the same wrapper problem, differently, and
usually without the same care.

## What it does

- A canonical `NatsClient` wrapper: `connect()`, `shutdown()`, `ping()`.
- Typed `Subject` objects instead of raw strings.
- `BaseModel`-only publish (no untyped payloads).
- Keyword-only subscribe, closing off a real production bug class where a
  callback gets silently mistaken for a queue group (nats-py 2.10+).
- Default dead-lettering on uncaught subscribe exceptions.
- JetStream KV bucket helpers, used as the L2 tier by `core`.

## Design philosophy

A single canonical wrapper so platform services and any 3tears app never
need to depend on a host repo just for NATS primitives, and never
reimplement the same mistake-prone raw-client patterns. The API is
deliberately "mistake-proofed": choices like keyword-only subscribe and
typed publish exist specifically to close off failure modes that have
actually happened in production, not as abstract hygiene.

## When to adopt

Any multi-pod deployment of `core` (as the L2 client), or any app that
talks to NATS directly and wants a safer wrapper than the raw client.

## Composes with

- [`core`](core.md) -- consumes this as the L2 cache client.
- [`epoch`](epoch.md) -- uses it for config-epoch broadcast.
- [`registry`](registry.md) -- uses it for tool call routing.

## Install

```bash
pip install 3tears-nats
```
