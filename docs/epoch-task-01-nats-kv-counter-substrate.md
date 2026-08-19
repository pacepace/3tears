# epoch-task-01: Move the ephemeral epoch counters from Postgres to NATS KV

**Status:** READY, but **not independently shippable**. See "Shipping order".
**Scope:** `3tears-epoch` (`client.py`), `3tears-nats` (`subject_permissions.py`),
`3tears-mcp` (integration tests), enforcement tests.
**Depends on:** nothing to build. Blocked from shipping alone by epoch-task-02.

---

## Objective

`EpochClient` keeps its counter in a Postgres `config_epochs` row. Move the *ephemeral*
epochs onto NATS KV so `bump` and `current` stop touching L3, and leave the one epoch
family that genuinely needs durability where it is.

## Why

The epoch is a cache-coherence signal, not a durable fact about data.
`EpochListener._last_seen` is process-local and resets on restart, so if every pod
restarts they prime against cold caches whatever the stored number says.

The L3 load is real but smaller than an earlier draft of this doc claimed. `echo()` short
-circuits without a read when the echoed value is not ahead (`listener.py:313-315`), so
the cost is one read per pod per subject per tick, plus one per genuine bump. Not
proportional to traffic. That is still a Postgres round trip on the coherence path for a
signal that has no business being durable.

## Reuse: do not build a counter, adopt `DistributedCounter`

`packages/core/src/threetears/core/coordination/distributed_counter.py` is already
"atomic increment/decrement counter over NATS JetStream KV". `_ensure_bucket` (`:262-268`)
opens with `storage="memory", create_if_missing=True, history=1` — exactly the substrate
this task specifies. `increment(key)` returns the new **per-key** value through a bounded
CAS loop.

Adopting it deletes a problem an earlier draft invented. Using raw KV stream revisions
would have made epochs bucket-global and sparse, breaking `current()==0` meaning "never
bumped" and "the first bump returns 1", and forcing every consumer to absorb that.
`DistributedCounter` keeps per-key contiguous counters, so those semantics survive
untouched.

The tradeoff to accept knowingly: a CAS retry loop under contention rather than a single
write. Epoch bumps are administrative and low-rate per subject, so this is the right side
of that trade. `3tears-epoch` already depends on `3tears` (`packages/epoch/pyproject.toml:24`),
so the dependency direction is unchanged.

Other reuse targets, all present, none to be reimplemented:

- `KvCapable` / `KvBucketLike` Protocols at `packages/nats/src/threetears/nats/kv.py:437-490`.
  Every KV consumer in the repo takes `KvCapable` plus a bucket name and binds lazily
  (`lease.py:349`, `token_bucket.py:400`, `idempotency.py:372`). `EpochClient` already
  holds a `NatsClient` for its broadcast (`client.py:112`), so it opens its own bucket
  rather than being handed one.
- `FakeKvBucket` / `FakeNatsClient` at `packages/core/src/threetears/core/testing/kv.py:59,209`
  are the test doubles. Do not write new ones.

## Carve-out: `datasource_tile_epoch` stays durable

**This epoch is not ephemeral and must not move.** `subjects.py:1857-1866` states the
value it carries is the `v{n}` segment of a tile URL, and
`packages/geo/src/threetears/geo/collection.py:255` puts that version in the cache key.
The number therefore escapes into browser and CDN caches that this system cannot reach.

A memory-backed bucket resets on a NATS restart, which would re-issue `v1..vN` for
different content while edge caches still hold the old generation keyed on the same
version. That is a correctness failure outside our blast radius, and no amount of
in-process detection fixes it.

So `config_epochs` survives for this family. `EpochClient` gains an explicit substrate
choice rather than a wholesale migration, and the durable path stays the default for any
epoch whose value is published outside the cluster. **Record in this document which
subjects take which path before building.** Long term the tile version is arguably a
content version rather than an epoch and belongs elsewhere entirely; that is separate
work, not this task.

## Key derivation, and the two ways it bites

`Subjects._sanitize` (`packages/nats/src/threetears/nats/subjects.py:160-172`) replaces
`.` with `-` and nothing else, while nats-py enforces `^[-/_=\.a-zA-Z0-9]+$` on KV keys.
An earlier draft's claim that no sanitisation is needed was wrong in two ways:

1. **Arbitrary caller strings.** `Subjects.datasource_tile_epoch(datasource_id, layer)`
   takes a free-form `layer`. A layer named `"census tracts"` is a legal Postgres PK today
   and an `InvalidKeyError` tomorrow, raised at `bump()` in production. Either validate
   the path and raise a typed error, or derive the key through the existing
   `_digest_token` (`subjects.py:175-186`), which exists for exactly this problem.
2. **Wildcards.** `listener.py:193` calls `current()` for wildcard subjects and the
   wildcard-priming path documented at `listener.py:139-173` depends on getting `0` back.
   Under Postgres a wildcard path simply matches no row. Under KV, `*` and `>` are illegal
   key characters, and `InvalidKeyError` is not in `get_entry`'s passthrough tuple
   (`packages/nats/src/threetears/nats/kv.py:316`), so it triggers a pointless `_reopen()`
   and surfaces as `KvError`, killing `subscribe()`. `current()` must return `0` without
   touching KV when the path contains `*` or `>`.

## Broker grants (silent failure if missed)

The new bucket needs a `kv_buckets` entry for `Principal.AGENT_POD`, `Principal.HUB` and
`Principal.GATEWAY` in `packages/nats/src/threetears/nats/subject_permissions.py`
(existing tuples at `:263`, `:525`, `:591`). Epoch *subjects* are already granted; the
buckets are not.

`tests/enforcement/test_kv_bucket_grant_naming.py:9-16` records why this matters: a
missing grant blocks to its deadline rather than raising, indistinguishable from an
unreachable broker. Here that is a 10s hang per `current()`
(`kv.py:59`, `_KV_OP_TIMEOUT_SECONDS`). Name the bucket suffix in this doc, and add the
paired assertion in the style of
`test_the_lease_bucket_a_tool_pod_is_granted_is_the_one_kvlease_opens`.

## Migration retirement

`docs/how-to-add-a-migration.md:198-200` forbids editing an applied migration: "Add a new
migration to evolve." With the carve-out above, `config_epochs` is not dropped at all, so
`migrations/` and `register()` stay. That also keeps the out-of-repo hub's
`build_platform_runner` import working, which deleting `register` would have broken at
import time.

## Shipping order

Task-01 must not ship without epoch-task-02. On a broker restart, `_reopen`
(`kv.py:217-236`) recreates the bucket empty and every operation *succeeds*, so `current()`
returns 0 while a surviving `EpochListener._last_seen` still holds 500. `catch_up`'s guard
is `if current > last_seen` (`listener.py:274`), so the callback never fires again for the
life of the process. Postgres has no equivalent mode. Either ship the two together, or
add the interim guard here: treat `current < last_seen` as a generation reset and
re-dispatch.

## Consumers (verified, correcting an earlier draft)

`3tears-channels` does **not** depend on `3tears-epoch` and never constructs a client; its
only references are prose (`websocket.py:1045`, `presence/wire.py:19`). `3tears-mcp`
depends on it (`packages/mcp/pyproject.toml:25`) but injects rather than constructs
(`auth.py:410-411`), and `rbac.py:12-14` says explicitly it holds no client reference.

The actual in-repo construction sites are tests:
`packages/epoch/tests/integration/test_multi_pod.py:148-150,197-198,247-248,287-289` and
`packages/mcp/tests/integration/test_multi_pod_rbac.py:187,251,307,320,359,366`. The
production construction site is the out-of-repo hub.

## Tests that will break (do not claim "unchanged")

- `packages/epoch/tests/unit/test_client.py:57,117,119` assert on the SQL strings.
- `packages/epoch/tests/integration/test_multi_pod.py:33,74,104-105` and
  `packages/mcp/tests/integration/test_multi_pod_rbac.py:45` import the migration and
  build schemas containing `config_epochs`.
- mcp *unit* tests are fine; `test_auth.py:21-34` uses `MagicMock`.

## Prose to update

`packages/epoch/README.md:27,42,50` documents the DDL and the `ON CONFLICT` SQL verbatim.
Module docstrings describing Postgres as source of truth: `client.py:1-23`,
`listener.py:86,144,215`, `wire.py:33`. `EpochClient.current`'s docstring, which promises
`0` means "nobody has bumped this domain in this database".
`packages/nats/src/threetears/nats/subjects.py:1802-1803,1866` and
`packagesges/nats/tests/unit/test_subjects.py:320-326`, which calls the subject path a
"row PK". `tests/enforcement/test_cache_primitive_usage.py:93-102`, whose `config_epochs`
allowlist entry narrows to the carve-out. Root `CHANGELOG.md`.

## Version

This changes `EpochClient.__init__` and is a breaking API change on a family at `0.26.1`.
Per `CLAUDE.md` the whole `3tears*` family moves to `0.27.0` in lockstep and every
intra-family bound moves with it
(`tests/enforcement/test_intra_family_version_bounds.py`). `bump-version.sh` only rewrites
bounds that existed when it ran, so grep `0\.26\.` after the bump.

## Acceptance

- `EpochClient` uses `DistributedCounter`; no new counter, no new KV wrapper, no new
  test double.
- `current()` returns `0` for wildcard paths without touching KV.
- A non-alphanumeric layer name round-trips through `bump()` without `InvalidKeyError`.
- The bucket appears in all three principals' `kv_buckets` with a paired enforcement
  assertion.
- `datasource_tile_epoch` still resolves through the durable path, proven by a test, not
  by a comment.
- Family at `0.27.0` with no `0.26.` bound left anywhere.
