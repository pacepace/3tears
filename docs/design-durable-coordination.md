# Durable coordination without promoting L2

`(3tears)`

## The rule

**Every NATS KV bucket is memory-backed.** NATS is L2. Durability is L3's job, reached
through `BaseCollection`, which already puts L1 and L2 in front of L3 so hot reads never
touch the database.

Enforced by `packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`. When the
last file-backed site in this repo is gone, the gate moves to `threetears.enforcement`,
consumer repos adopt it, and it loses its exemption mechanism: file-backed KV then cannot
be exempted, only designed out.

## Why file storage is the wrong answer even where it works

On cobalt-dev file-backed JetStream genuinely persists -- three bound 20Gi EFS volumes,
`store_dir: /data/jetstream`. The objection is architectural. A file-backed bucket is the
cache tier taking the source-of-truth role while keeping none of the properties that role
needs:

- **`num_replicas=1`.** It survives a pod restart because the volume reattaches. It does
  not survive losing the node or the volume.
- **No backups.** L3 is backed up. A JetStream volume is not.
- **No schema, no migrations.** Long-lived state acquires a shape; KV gives it none.
- **Storage is chosen at CREATE and never reconciled.** A wrong value is fixed by deleting
  live state on a running cluster, not by shipping a fix.

`threetears.epoch` made the same call for its durable tile family: file storage survives
only while its store directory survives, so it is a false guarantee against the failure it
claims to cover.

## Every file-backed site, classified by what a wipe must do

The four `threetears.core.coordination` sites were never one kind of state, so they do not
get one answer. Each was classified from its production consumers.

### Single-use nonces: memory, fail closed by watermark

A nonce only matters inside its accept window, and it never had to survive a wipe. It had
to make a wipe fail CLOSED. The former file-backed design did not even do that: losing the
JetStream volume reopened the replay window.

`ReplayGuard` stays on memory storage and learns when its bucket was created:

1. `record_unique(nonce, *, issued_at)` creates the nonce key (create-if-absent). A
   present key is a replay, refused.
2. On a fresh create, it reads the stream's creation time from the server
   (`STREAM.INFO`, `StreamInfo.created`), and refuses the artifact when
   `issued_at < created + verifier_future_tolerance + CLOCK_DRIFT_ALLOWANCE`.

**Why the order is create, then read.** A bucket handle holds only names, so after another
pod recreates a wiped stream a stale handle keeps working silently -- a creation time
cached at open is unsound, and reconnect callbacks run after new requests can already be
served. Reading after the create closes both: any wipe before the create is visible as a
newer creation time, which can only make the check stricter. A replay can only create its
key if the original entry was wiped, which puts the original acceptance, and so its signed
issue time, before the new creation time.

**The `issued_at` contract:** no later than the earliest moment the artifact could first
have been accepted. Each call site derives it from a signed or server-held timestamp:
DPoP and PoP `iat`, the proxy assertion `iat`, SAML `IssueInstant`, the survey challenge's
HMAC-signed issue time, and for an OAuth client assertion whose `iat` is optional,
`exp - MAX_CLIENT_ASSERTION_LIFETIME`.

**How far the refusal reaches.** A verifier accepts an issue time up to its future tolerance
ahead of its own clock, and the creation time is the broker's clock. So the refusal reaches
the verifier's future tolerance, passed at construction with no default, plus a single
named drift allowance between those hosts, added by the guard. Each verifier calls
`require_covers` with its own leeway, so widening a leeway later fails at startup instead of
silently reopening the hole. The cost is bounded and visible: for that long after a wipe,
fresh artifacts are refused. A missing `created` raises rather than admits.

**Some guards are removed rather than watermarked.** Where the guarded artifact is itself a
server-side record read before the nonce is recorded -- OAuth authorization codes, OIDC and
GitHub state, SAML `InResponseTo`, passkey challenges, TOTP partial-auth -- the artifact is
consumed by a revision-guarded delete of its own record, as `NatsKvTicketStore` already
does. Separate R1 streams can sit on different NATS nodes, so a separate nonce bucket can be
wiped while the artifact survives. Consuming the artifact itself cannot split that way.

### Durable security state: L3 through `BaseCollection`

| State | What a wipe costs | Write path |
|---|---|---|
| Refresh-token jti ledger | an already-rotated refresh token is accepted again, for its whole 30-day life, and reuse detection is skipped | synchronous L3 |
| Standing revocations (`sid`, `sub`, `customer_id`) | revoked sessions become valid | synchronous L3 |
| Attempt counters (`WindowedCounter`, lockout, spray) | lockouts release; brute-force budgets restart | L2 CAS, write-behind to L3 |
| Idempotency claims | a retried operation runs a second time | L2 claim, write-behind to L3 |

**Counters are write-behind on purpose.** A NATS wipe coinciding with a crash of the writing
pod loses at most one flush interval of increments. That is a few extra attempts against a
throttle, and it is not worth a database write on every login or API call. Revocations lose
nothing.

### Hot-path cost

- **Revocation checks:** an L1 hit in steady state. Other pods see a revocation through the
  invalidation broadcast.
- **Counter increments:** L2 CAS, the NATS round trips they already pay, with L3 batched per
  flush interval.
- **Nonces:** memory plus one extra NATS request on a fresh nonce. No L3.
- **L3 sees:** one miss per key per L2 lifetime, batched flushes, and rare revocation writes.

These are measured, not assumed, before the primitives adopt them.

## What `BaseCollection` gains

Each gap is a generic enhancement to the primitive, not a store beside it.

1. **Negative caching.** Today a miss in every tier caches nothing, so every "not revoked"
   check would reach L3. On a full miss the collection writes an L2 absent-marker with
   create-if-absent, so a reader can never overwrite a writer's value and a writer's put
   replaces the marker. A collection that opts in makes its own L2 writes strict: a failed
   overwrite raises rather than leaving a stale "absent".
2. **Row expiry.** A declared expiry column every tier treats as absent once passed. A sweep
   is table-size hygiene only, never correctness.
3. **`l2_cas_mutate` on a three-tier collection.** It seeds from L3 when L2 holds nothing, so
   a wipe does not reset a counter to zero; persists the result through the collection's
   flush policy; and returns the outcome, so a claim can report claimed or exists.
4. **Flush policy per collection.** Today it is one global strategy plus a table-name list.
   A counter wants write-behind, a revocation wants synchronous L3.

## Consumers

The primitives keep their public surfaces apart from `ReplayGuard.record_unique` gaining
`issued_at`, which is a breaking change landed at every call site in one commit per repo.

- **identity** (the largest): the removals above, the watermark at DPoP, SAML assertion and
  client-assertion sites, jti and standing revocations and every counter (including its own
  `security/spray_counter.py`, which bypasses `WindowedCounter` and is file-backed) onto
  collections, its own migration, and a one-time copy of the live revocation buckets into
  the new tables before the new code rolls. Without the copy, revoked tokens become valid.
  identity-edge holds no database by design, so its fail-open route throttles run the same
  collection without an L3 pool.
- **hub**: the DPoP impersonation guard's `issued_at`.
- **registry and tool runtime** (this repo): the PoP and proxy-assertion guards.
- **survey**: the entry-challenge guard, the panel lockout counter, and idempotency claims.
- **scriob**: its login throttle.

**The live buckets are converted, not abandoned.** The nonce buckets keep their names and a
bucket's storage is never reconciled, so after release the new code binds the existing
file-backed streams. Each one stays file-backed until deleted by name, and a deletion is a
wipe: calls through that guard are refused for its reach while it is recreated
memory-backed. The durable primitives' buckets are different: once their state lives in
L3 they are genuinely unused, and they are deleted after the one-time copy. Both happen on
cobalt-dev and then prod, after every consumer is released, dry run first, and a real
sign-in verifies each.
