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

## KV bucket grants

A KV operation is a JetStream API request, and a request the server refuses
on permissions is never answered. It has no error to return, so the call sits
until the wrapper's 10-second deadline and reports a timeout, which is also
what an unreachable broker reports. The connection itself stays up and
carries every other subject perfectly well, so the obvious next step (check
the network) finds nothing wrong.

Declare each bucket a principal touches in
`threetears.nats.subject_permissions`:

```python
kv_buckets=(f"{ns}-epochs", f"{ns}-collections")
```

`mint_user_jwt` expands one entry into pub+sub on `$KV.{bucket}.>` and
JetStream control over the backing stream `KV_{bucket}`. Grant the data
subjects by hand and the control plane still refuses `STREAM.CREATE`, so the
bucket never opens.

The wrapper does the diagnosis for you rather than leaving it in this
document. The server announces a refusal once, as a `permissions violation`
frame on the error callback, and unlike an authorization violation it does
not close the connection, so nothing downstream re-reports it. That frame is
decoded to the bucket it names and logged with the `kv_buckets` entry to add.
The deadline path and a failed `kv_bucket()` open say the same thing.

`tests/enforcement/test_kv_bucket_grant_naming.py` compares grants against
openers **only where both live in the 3tears repository** -- it says so itself,
and it enumerates nothing. Your grants and your openers are not reachable by it.
Nothing checks that your `kv_buckets` entry matches the name your component
actually opens, and a mismatch produces exactly the timeout described above:
authorised against a bucket that does not exist while the one you open is
ungranted. Grant the name your opener produces, prefix included.

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
