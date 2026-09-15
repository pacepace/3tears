# Durable coordination without promoting L2

`(3tears)`

## The rule

**Every NATS KV bucket is memory-backed.** NATS is L2. Durability is L3's job.

Enforced by `packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`,
strict, with a temporary exemption per site listed below.

## Why file storage is the wrong answer even where it works

It is not that file-backed JetStream fails. On cobalt-dev it genuinely persists --
three bound 20Gi EFS volumes, `store_dir: /data/jetstream`, 54 days old. The
objection is architectural.

A file-backed KV bucket is the cache tier taking the source-of-truth role while
keeping none of the properties that role needs:

- **`num_replicas=1`.** It survives a pod restart because the volume reattaches.
  It does not survive losing the node or the volume, and while that one node is
  down the state is simply unavailable.
- **No backups.** L3 is backed up. A JetStream volume is not in that story.
- **No schema, no migrations.** A durable store people keep for months acquires a
  shape. KV gives it none, so the shape lives in whichever code last wrote it.
- **Storage is chosen at CREATE and never reconciled.** A wrong value is not a bug
  you fix by shipping a fix; it is one you fix by deleting live state on a running
  cluster. That asymmetry is the whole reason this is a gate.

`threetears.epoch` already refuses file storage for its own state, on the related
grounds that it would be a FALSE guarantee, and keeps a Postgres row instead. This
generalises that decision.

## The false choice

The reflex says: this state cannot be lost, so make the bucket durable. That reads
as a choice between speed and safety, and it is not one.

**`BaseCollection` composes both tiers.** L1 pod-local, L2 on this same NATS --
memory-backed, as intended -- and L3 in Yugabyte as the source of truth. State
that cannot be lost was never L2 state; it is L3 state with a cache in front, and
the platform's own primitive is exactly that shape.

## What has to move, and it is one change

Every file-backed site in this repo is in `threetears.core.coordination`, and all
four share one seam:

```python
self._bucket: KvBucketLike | None = None
async def _ensure_bucket(self) -> KvBucketLike:
    self._bucket = await self._client.kv_bucket(..., storage="file")
```

`KvBucketLike` is a **protocol**. So this is one new component and four one-line
adopters, not four rewrites.

| Site | What a wipe costs |
|---|---|
| `idempotency.py:372` | a retried operation runs a second time |
| `replay_guard.py:109`, `:264` | a code or token already spent is accepted again |
| `windowed_counter.py:320` | every lockout releases; every brute-force budget restarts |

None of those is a cache. That is the tell, and it is why they reached for file
storage rather than being careless.

## Shards

**Shard 1 -- the gate. DONE.** Walker and gate together in
`packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`, strict, with
the four sites exempted and each rationale naming the work that removes it. Stops
the next one; does not fix these.

**The walker is deliberately NOT in `threetears.enforcement` yet.** Every
violation today is in this repo, and promoting it would grow that package's public
API -- which on a patch line is refused by `test_api_growth_requires_a_minor_bump`,
for a real reason: the intra-family bound reads `>=0.41.0,<0.42.0`, so pip may
resolve a sibling published earlier on this line that lacks the new names, giving a
family that installs clean and ImportErrors at runtime. It moves when a second repo
needs it, and that move is a minor bump by itself.

**Shard 2 -- `DurableKvBucket`.** An implementation of `KvBucketLike` backed by a
`BaseCollection`. Same surface as the KV bucket the four already hold, so adoption
is the `_ensure_bucket` line and nothing else. TTL semantics are the design
question: KV expiry is the broker's, a collection's is the table's, and the
primitives lean on per-entry TTL rather than sweeping. Decide that before writing
it, not during.

**Shard 3 -- adopt it, four call sites**, and delete the four exemptions in the
same commit. The gate's staleness check fails if an exemption outlives its call
site, so this cannot be half-done quietly.

**Shard 4 -- identity's revocation lists**, which are not coordination primitives
and are the largest durable thing still in KV: `identity-revocation-jti` held
11,431 entries on cobalt-dev. Losing it makes revoked tokens valid again. Its own
collection, its own migration, in the identity repo.

## What this does not claim

Moving to L3 costs a round trip these primitives do not pay today. `Idempotency`
and `ReplayGuard` sit on hot paths. Shard 2 has to measure that rather than assume
it is acceptable -- and if it is not, the answer is a considered L1/L2 read-through
in front of an L3 truth, which is what `BaseCollection` already does, rather than
a return to file storage.
