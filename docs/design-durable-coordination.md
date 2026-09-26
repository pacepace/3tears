# Durable coordination without promoting L2

`(3tears)`

## The rule

**Every NATS KV bucket is memory-backed.** NATS is L2. Durability is L3's job, reached
through `BaseCollection`, which already puts L1 and L2 in front of L3 so hot reads never
touch the database.

Enforced by `threetears.enforcement.memory_only_kv`, which consumer repos import and run;
`packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py` is now a four-line shell
over it. The last file-backed site in this repo is gone, and the gate has **no exemption
mechanism**: `_kv_memory_only_exemptions.txt` and the code that read it were deleted, so
file-backed KV cannot be exempted, only designed out. The walker scans nested package families
too, and refuses to pass when its globs match nothing.

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
`require_covers` with its own leeway, so widening a leeway later fails loudly instead of
silently reopening the hole: at construction for the registry proxy and the tool server, and
on every request for `validate_dpop_proof`, which is a function with no construction step.

**Bind at start, or the first call after a bucket is created is itself refused.** The bucket is
created by whichever call opens it first. A guard left to open it in its first `record_unique`
creates it there and then compares that call's artifact against a creation time of a moment ago
-- inside the window by construction, however long after the service started the artifact was
issued. On 2026-09-25, after a Docker restart, that refused a login with a 65s reach (surfacing as
"invalid username or password") and a tool call with a 5s reach (surfacing as "proxy assertion
nonce replay"). So a service calls `ReplayGuard.bind()` at startup, before it serves anything:
the bucket is then created before any artifact the process could issue or accept, and after a
wipe the watermark refuses only what was issued before the service started, or within the reach
of its start. `ToolServer.serve` and `CallProxy.start` bind their guards before subscribing;
a guard a consumer builds itself, like the hub's DPoP guard and identity-core's, is that
consumer's to bind at its own startup. `record_unique` still binds an unbound guard, so
forgetting costs only the window.

**`bind()` stamps the anchor before the first nonce.** Left to the first record, the anchor is
read after that nonce's create. A wipe landing between the two, with another replica's record
stamping the anchor first, would put the ledger's birth after the wipe: the guard would read a
first run and admit a replay of that nonce. Stamped at bind, the ledger's birth predates every
nonce this process records. An anchor that cannot be read or written never fails the bind -- it
is logged, the guard keeps the conservative watermark, and the next record tries again. Only a
bucket that cannot be opened fails it.

**A wipe under a running process is recreated at reconnect.** The bound handle is kept for the
process's life, and the wrapper's self-heal recreates a vanished stream on the next operation
through it. Left there, the recreated bucket is younger than the restart by however long the
service sat idle, so the first artifact after an idle wipe is refused the same way. So `bind()`
also registers, once per guard, a hook on `NatsClient.add_reconnect_callback` that touches the
bucket after each reconnect, recreating it as the connection comes back. That is availability
only: correctness still rests on `record_unique` reading the creation time fresh after every
fresh create, so a wipe at any moment, hooked or not, can only make the check stricter -- which
matters because reconnect callbacks run after new requests can already be served. The hook
never raises into the reconnect path; a failed touch is logged and the next record recreates the
bucket as before. The client has no way to remove a hook, and a guard does not need one: both
live as long as the service. A client without `add_reconnect_callback` (any other `KvCapable`)
binds as before without the hook. A wipe while NATS stays up -- a stream deleted by hand, with no
reconnect -- is still recreated by the next use. In production the bucket outlives every pod
restart (it lives with the broker), so what remains is paid once per broker restart, not once
per pod. A test standing up a fresh namespace pays it every run unless it binds first, which is
why `test_tool_server_nats.py` warms the bucket and waits the window out rather than pretending
the first call should succeed. The cost is bounded and visible: for that long after a wipe, fresh
artifacts are refused. A missing `created` raises rather than admits.

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

**Counters are write-behind on purpose.** A NATS wipe loses at most the increments made since
the last flush: the next increment starts from L3, which holds the last flushed count, not
from any pod's unflushed buffer. That is a few extra attempts against a throttle, and it is
not worth a database write on every login or API call. Revocations lose nothing. A
write-behind collection needs something to drive `flush_pending` on an interval. The
COLLECTION owns that, not the primitive: `CoordinationCollection.l2_cas_mutate` and
`save_entity` arm a `threetears.core.coordination.flusher.PeriodicFlusher` on the first write,
so a wave-2 primitive cannot forget it. Wire no flusher of your own; the one thing a consuming
process must do is call `CollectionRegistry.close_collections()` in its shutdown path, or the
last interval's rows are lost.

### Hot-path cost

- **Revocation checks:** a revoked key is an L1 hit in steady state, and other pods see a new
  revocation through the invalidation broadcast. A key that is NOT revoked -- nearly every
  check -- is answered by an absent-marker in L1 (or L2, on a pod that has not asked yet),
  after one NATS read of the table's write generation: what the KV-backed check costs today,
  and no L3 query.
- **Counter increments:** L2 CAS, the NATS round trips they already pay, with L3 batched per
  flush interval.
- **Nonces:** memory plus one extra NATS request on a fresh nonce. No L3.
- **L3 sees:** one miss per key per pod per write generation (fewer where L2 already holds
  the marker), batched flushes, and rare revocation writes. The generation is per table, so
  this is cheap only for tables written rarely; see "What one generation per table costs".

**Measured, not assumed.** `scripts/measure-coordination-latency.py` runs each hot operation on
both paths against one broker and one database, 200 rounds each. On a 2026-09-16 developer
machine with containerised NATS and Postgres:

| Operation | Path | Median | p95 | L3 reads |
|---|---|---|---|---|
| counter increment | bare KV (the old path) | 0.805 ms | 1.717 ms | -- |
| counter increment | collection | 0.924 ms | 1.886 ms | 0 |
| revocation check, not revoked | bare KV (the old path) | 0.388 ms | 0.618 ms | -- |
| revocation check, not revoked | collection | 0.455 ms | 2.187 ms | 50 per 200 checks |
| revocation check, revoked | collection | 0.042 ms | 0.055 ms | 0 |

What the numbers say. A counter increment costs about 0.12 ms more than the bucket it replaced,
which is the price of durability on a path that already paid for a NATS round trip. A check of a
key nobody revoked costs about 0.07 ms more at the median; the 200 checks covered 50 distinct
keys and made 50 L3 reads, one per key -- every later check was a marker hit, which is the claim
negative caching exists to make. A check of a key that IS revoked is an L1 hit at 0.04 ms and no
L3 read at all. Re-run the script rather than trusting this table.

## What `BaseCollection` gains

Each gap is a generic enhancement to the primitive, not a store beside it.

1. **Negative caching** (`negative_cache_max_age`), **stamped with a write generation.** Today
   a miss in every tier caches nothing, so every "not revoked" check would reach L3.

   *How it answers.* A reader reads the table's write generation BEFORE its L3 lookup; a full
   miss is recorded as an absent-marker in its L1 and in L2 under that generation. A marker
   answers only while its stamp is the current generation. Every committed write advances the
   generation, so a marker recorded from an L3 read that predated a write carries a
   generation the write already moved past.

   *Why a generation and not timing.* The first design wrote an L2 marker create-if-absent
   and trusted the invalidation broadcast to clear it. Review found three ways that lies: a
   peer's listener deletes the writer's fresh L2 value and the marker is created over the
   empty slot; a writer in another principal broadcasts before the reader's marker exists; and
   a marker timestamp trusts the writer pod's clock. A generation comparison is immune to all
   three -- it does not care what L2 holds, who wrote, or when anything was delivered.

   *Where the generation lives.* `threetears.epoch`'s memory bucket, one value per table,
   `"{incarnation}:{count}"`. One value, because reading an incarnation and a count as two
   keys can straddle a broker wipe and produce a token a later genuine one can equal. A wipe
   forces a new incarnation, so no token issued before it matches one issued after. Core
   defines the `GenerationSource` protocol; `EpochGenerationSource` implements it. The read is
   only as fresh as the replica that answers it: the epoch bucket is single-replica today, so
   every read reaches the writer's copy. Replicating that bucket with direct gets enabled would
   let a follower answer with a generation a write had already advanced past, and has to be
   designed for rather than switched on.

   *What bounds memory, for rows as well as markers.* A row on a table that declares
   `expires_at_column` carries that expiry into L2 as a server-side lifetime on every write, so
   the broker reclaims it without being asked. Without that, L1 dropped an expired row on read
   and the owning collection swept L3, while L2 -- the shared, memory-backed collections bucket,
   opened with no expiry -- kept every retired key until the broker restarted.

   *What bounds marker memory.* An L2 marker carries a server-side per-entry lifetime of the max age
   (NATS 2.11+ `allow_msg_ttl`, reconciled in place on buckets created before it was set), so
   markers nobody reads again leave the shared bucket. An L1 marker lives in a framework-owned
   table beside the collection's and stops answering at the same age; a sweep, run from the
   marker write path at most once a minute, drains every past-deadline row in batches, so the
   table holds at most one max age plus one minute of markers whatever the miss rate. Most rows
   need it: a denylist checks one key per token, and nobody looks those keys up again.

   *What one generation per table costs.* Any committed write to the table stops every marker
   in it from answering, not only the written key's. Every writer also compare-and-swaps the
   same generation key, so heavy concurrent writing contends on it. After 30 lost rounds the
   writer raises `GenerationUnavailableError`, with the row already committed. So negative
   caching suits tables that are read far more often than they are written, like standing
   revocations. The refresh-token jti ledger is not one of them: it is written on every
   rotation, and each check is followed by that write, so a marker would rarely answer twice.
   Per-key generations would lift this and are not built; a table that needs them is the
   reason to build them.

   *Who must advance.* Every writer that opts in and has an L3 pool, whether or not it has an
   L2 client of its own. Absences are recorded by readers, which may be other pods with L2; a
   writer skipping the advance for lack of L2 would leave their absences answering. The
   guarantee also holds only for writes made through the collection's own write paths
   (`save_entity` and `l2_cas_mutate`). An L3 write that bypasses them advances nothing:
   ad-hoc SQL through `l3_pool`, or a subclass that writes its own SQL and then fills L2. So a
   collection that caches absences writes only through those paths, and
   `packages/core/tests/enforcement/test_negative_cache_write_paths.py` is the structural check
   that holds it. It resolves base classes within one module, so a cross-module subclass of a
   negative-caching collection is out of its reach; nothing in the repo is in that shape, and
   the walker's own docstring says so.

   *Which writes need no advance, and why that is not a loophole.* A marker only ever claims a
   key is ABSENT, so a write that cannot make a key present cannot make a marker wrong. `delete`
   is one. So is the expired-row sweep the coordination tables ship: every tier already reads an
   expired row as absent, so removing it changes no answer. Those two are the whole list, and
   the structural check permits exactly that shape. Encode a new shape there rather than
   widening the check to fit whatever code it meets.

   *What still needs the max age.* A write that commits and then fails to advance the
   generation leaves older markers answering; the writer raises, and the max age bounds the
   window if nobody retries. For the same reason opting in refuses deferred flushes, subscript
   writes and joining a caller's transaction, each of which could make a row visible before
   the generation moves.

   *What does not need it.* A failed L2 write degrades exactly as it does on any collection. The
   marker it failed to replace was stamped before the commit advanced the generation, so it
   stops answering anyway.
2. **Row expiry** (`expires_at_column`). A row whose declared expiry has passed is absent to
   every read that answers "does this exist" -- `get`, `ensure`, `collection[id]` -- at L1, L2
   and L3. Reporting reads that serve an entity's own internals still see it, so an entity
   held past its expiry can still be saved. The L3 check runs on the fetched row, so any
   `fetch_from_store` inherits it; a collection may also filter in SQL. A sweep is table-size
   hygiene only, and the COLLECTION drives it, beside the flusher: `l2_cas_mutate` and
   `save_entity` call `sweep_expired_if_due` after every write, which is self-throttled to once
   per interval per process and contains its own failures, so a failed sweep never fails the
   throttle or claim that triggered it. It is not each primitive's to remember, for the same
   reason the flusher is not: four primitives already disagreed about where to call it.
3. **`l2_cas_mutate` on a three-tier collection.** The L2 revision stays the concurrency
   fence; L3 becomes the durable record behind it.
   - When L2 holds no live row, the callback is shown L3's row, so a wipe does not reset a
     counter to zero. There is no separate seeding write: replicas racing to seed all try
     create-if-absent, one wins, and the rest retry against its value.
   - Only a won result is persisted, so a write that lost the race never reaches L3.
   - It returns a `CasMutation` (created, updated, deleted, noop), so a claim can report
     claimed or exists.
   - A collection that caches absences advances its write generation after the persist,
     exactly as `save_entity` does.
   - A persist that fails withdraws the won L2 value before the error propagates. The value is
     deleted at the revision it won rather than restored, because with L3 behind it an absent
     key is always correct. Without that, a retried claim reads "already claimed" and skips
     its work, and a retried increment counts twice. A table that fences every L3 write
     (`cas_null_safe`) is refused before L2 is touched, since its unfenced persist would fail
     every update.
   - *Peers keep the key.* The invalidation broadcast from a compare-and-swap names the L2 scope
     it left current, and a listener in that scope skips its L2 eviction. That key is the
     fence, and under write-behind the only copy newer than L3; deleting it would move the
     counter backwards. Listeners in other scopes still evict their own copies. An unfenced
     `save_entity` broadcast still evicts everywhere, because its put can land out of order.
   - *One principal per row.* The fence is a scoped key, so two principals mutating one row
     would hold two fences and overwrite each other in L3.
   - *What L3 does not order.* Two replicas that win consecutive revisions persist
     independently, so L3 can briefly hold the earlier row. L3 is read only once L2 has lost
     the key, so insert-or-delete data (revocations, the jti ledger) is unaffected and a
     counter can at worst resume a few increments low.
4. **L3 write policy per collection** (`l3_write_policy`). The process-wide strategy and table
   list stay the default; a collection that knows what its data can tolerate declares
   `"write_behind"` (counters) or `"synchronous"` (revocations), and the declaration wins. A
   write-behind declaration without a write buffer is refused at construction, and so is one
   on a collection that caches absences. Deletes always land synchronously, because the
   buffer holds rows, not removals.

## The tables the primitives keep their state in

`BaseCollection` is the machinery; these four tables are what the primitives put in it. All in
`threetears.core.coordination.tables`, all keyed `(purpose, key)`.

| Table | Holds | Policy |
|---|---|---|
| `coordination_counters` | the count in the live window, and when it opened | write-behind, expiring |
| `coordination_claims` | an idempotency claim and its stored outcome | write-behind, expiring |
| `coordination_revocations` | the moment a key is revoked from | synchronous, negative-cached, expiring |
| `coordination_redemptions` | a single-use redemption, presence only | synchronous, expiring |

- **`purpose` is what a bucket name used to be.** One process builds many primitives over one
  table -- identity-core has three counters, identity-edge seven route throttles -- and a
  collection is registered per table, so a table per bucket would mean seven collections and
  seven migrations for one process. The rows stay separated by `purpose`, and
  `coordination_collection` hands every primitive the one shared collection.
- **Which tiers are optional depends on what the primitive promises.** identity-edge has no L3 by
  design and still throttles across replicas on L2 alone; a registry with no L2 counts per
  process. A collection with no L3 reports its writes as done rather than raising, because there
  was nothing to write to, and logs that once per table so a wiring gap is not mistaken for the
  deliberate case. But a primitive whose contract is exactly-once -- the redemption ledger, the
  idempotency store -- refuses a registry with no L2 at construction: the compare-and-swap is the
  guarantee, and without it two replicas can both be told they were first.
- **One declaration, two readers.** The migration renders its DDL from the same `TableSchema` the
  collection reads, so a consumer's table cannot drift from the table the code expects. A
  consumer that cannot run DDL -- an agent or tool pod, whose broker refuses it -- declares the
  same schemas in its data section instead.
- **Lifecycle.** A write-behind table's flusher starts on its first write and is stopped by
  `CollectionRegistry.close_collections()` in the process's shutdown path, beside
  `stop_invalidation_listener()`. Without that call the last flush interval is lost on shutdown.

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
- **hub**: its DPoP guard (`hub-dpop-nonces`, built in `aibots/hub/app.py`) gains
  `verifier_future_tolerance` covering the `iat_window` it validates with. Without it the hub
  fails at startup on this release. `validate_dpop_proof` passes `issued_at` itself.
- **identity's refresh-token jti ledger is not a nonce guard.** It was a `ReplayGuard` over a
  30-day TTL in a bucket of its own. Watermarked, a broker wipe would refuse every outstanding
  refresh token for up to 30 days, which is why it could not be one. It is an L3 collection
  now, a `purpose` named `identity-revocation-jti` rather than a bucket of that name -- the
  move this bullet used to describe as pending. The bucket is decommissioned, and the name was
  dropped from `threetears.iam.buckets` in 0.45.1 so the platform stops granting it.
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
