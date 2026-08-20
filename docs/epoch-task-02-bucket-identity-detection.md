# epoch-task-02: Detect a recreated bucket and reset, instead of dropping bumps forever

**Status:** BUILT, not shipped -- no PR, not merged, not released. Landed as a series of commits on the branch that carries this file; `git log --oneline
-- packages/epoch` is the current answer, and a range written here goes stale the next time
any of it is touched. The in-process reopen signal below is DESCOPED; see that section. Reshaped after review, which found the first draft could notify only one
consumer and could re-wedge itself.
**Scope:** `3tears-epoch` (`listener.py`, `client.py`), `3tears-nats` (`kv.py` and the
`KvBucketLike` Protocol at `kv.py:437-464`), `3tears-core`
(`testing/kv.py`, to keep `FakeKvBucket` in parity).
**Depends on:** epoch-task-01. Must not release without epoch-task-03; see "The trigger".

---

## Objective

Give `EpochListener` a way to tell "the counter went up" from "the counter is a different
counter", and reset local state rather than silently dropping bumps in the second case.

## Why

A memory-backed KV bucket is wiped when NATS restarts, and its counter goes back to zero.
Pods do not restart when NATS does, so a live pod keeps `_last_seen` at, say, 5000, and
`listener.py`'s dedupe drops every subsequent bump as already-seen. Permanently. Every
cache served by that pod goes stale with no recovery path and no error.

The evidence is a shipped self-heal path and its test, not an anecdote:
`packages/nats/src/threetears/nats/kv.py:217-236` (`_reopen`) exists because a single-node
restart on ephemeral JetStream storage wipes every stream, and
`packages/nats/tests/integration/test_kv_self_heal_round_trip.py:23-44` reproduces it
against a real broker with one line, `await js.delete_stream(f"KV_{bucket.name}")`.

## Detection, in two independent halves

### 1. The in-process signal -- DESCOPED, deliberately

**Not built.** The identity key below is sufficient for correctness, and this half is an
optimisation on detection LATENCY for exactly one pod: the one whose own operation triggered
the reopen. That pod learns from the identity check on its next catch-up pass like everyone
else, so nothing is permanently missed.

What it costs to skip: that pod's own bump broadcasts the new counter's first value, every
peer drops it as stale, and the correction waits for a catch-up pass rather than arriving
immediately. Acceptable because the pass is the mechanism the whole design already relies on
for a dropped broadcast.

Recorded rather than silently omitted because the shape below reads as two halves, and a
reader finding one implemented would otherwise assume the other was forgotten. If detection
latency ever matters, this is where it comes from, and the design is:

`_run_with_reopen` (`kv.py:238-257`) is the single funnel every KV op passes through
(`get` `:298`, `get_entry` `:316`, `put` `:338`, `create` `:355`, `update` `:386`,
`delete` `:427`). When it reopens, that process *knows* the bucket was recreated, and
today tells nobody.

Raise it. Follow the existing registration idiom rather than inventing one:
`NatsClient.add_reconnect_callback` / `ReconnectCallback` at `client.py:134-137`, dispatched
at `:1060-1070`.

This half matters because of an interleaving the first draft missed entirely: pod A's
`bump` hits a dead handle, `_run_with_reopen` recreates the bucket, the retried write
succeeds at the new counter's first value, and `EpochClient.bump` broadcasts it. Every
peer drops it as stale, and so does pod A. Without this signal nothing notices, even though
the reopen happened in-process.

### 2. The identity key (covers every other pod)

The bucket carries its own identity under a reserved key, `_bucket_identity`.

- **Every opener unconditionally attempts `create(key, uuid7())`, on every open and after
  every reopen.** Never `put`. The return value distinguishes winner from loser
  (`kv.py:343-370`: `int` on success, `None` on conflict), so no branch on "did I create
  the bucket" is needed. That branch would be wrong anyway: `open()` at `kv.py:164-196`
  treats a successful `STREAM.CREATE` as a create, and nats-server answers successfully
  when the config is identical, so the create branch is taken on essentially every open of
  an existing bucket. An implementer who gated a `put` on it would have every pod start
  rewrite the identity and flush the whole fleet.
- **Reading no identity is NOT a reset** -- an earlier draft of this section said the
  opposite, and the shipped code is the better rule. The identity is minted create-if-absent
  on every read, so a bucket that genuinely lost its key gets a NEW one immediately and the
  comparison catches it. The only way to read nothing is for KV itself to be unreachable,
  and an outage is not a replacement: treating it as one flushes every cache in the fleet on
  a blip, and forgetting the recorded identity would make the next successful read look like
  a change too.
- **The loser path is not atomic.** `create()` returning `None` then `get()` returning
  `None` (the bucket was wiped in between) is "identity unknown", which under the rule
  above is a reset. Retry the create/get pair a bounded number of times before concluding.

## What a reset does

**Clear, do not re-prime.** The first draft said "re-prime from the current revision",
which re-creates the bug. Identity and counter are two separate reads (`kv.py:285`,
`kv.py:306`) with no atomicity between them, so a counter value read against the old
generation can be written into `_last_seen` against the new one, wedging the pod again with
no further identity change to rescue it.

So: `_last_seen.clear()` and nothing else. Priming exists only to suppress a redundant
first callback on cold start; after a reset the redundant callback is the *desired*
outcome. Record the new identity only *after* the clear completes, so a third generation
arriving mid-reset triggers another reset rather than being swallowed.

## The listener must hold its callbacks (the first draft could not do this at all)

`EpochListener`'s only state is `self._last_seen: dict[str, int]` (`listener.py:79`).
`on_bump` is a closure argument to `subscribe` (`:98`) and to `catch_up` (`:247-251`),
captured by the inner `_on_bump` and never stored.

So "invokes the consumer callback" was not implementable. A pod subscribed to two subjects
that detects a reset while checking the first can invoke only that subject's callback. The
second's `_last_seen` is cleared and its consumer is never told, so its cache stays stale
until that subject happens to bump again, which for a quiet subject is never.

`subscribe()` must record `(Subject, BumpCallback)` in listener state, and a reset must fan
out to **every** registered callback. A wildcard registration gets one invocation for the
wildcard, not one per matched path.

## The reset signal must not be a bump

`BumpCallback` is `Callable[[int, dict | None], Awaitable[None]]` (`listener.py:41`). After
a reset the only epoch available is below everything the consumer has already acted on, and
the framework actively teaches consumers to dedupe on that number (`listener.py:1-12`, and
`echo()` at `:313-316` returns early on `echoed_epoch <= last_seen`). A consumer following
that guidance would discard the reset, reproducing the bug one layer up.

Define a separate `on_reset()` with no epoch argument, documented as never epoch-deduped.
It also keeps the listener out of cache vocabulary, which epoch-task-03's constraint
requires: the listener delegates, it does not know what a cache is.

## The trigger

**Do not release this without epoch-task-03.** In the window where 02 has landed and 03 has
not, the only thing calling `catch_up` anywhere is one consumer's hand-rolled loop at
`packages/mcp/src/threetears/mcp/auth.py:537-563`. Every other consumer would ship the
detection mechanism with nothing invoking it.

If they must ship separately, the interim trigger is checking identity inside
`EpochClient.current()`, so every existing `catch_up` and `echo` path carries it.

## Scope decisions to make explicit

- **The identity key is epoch-only, not a generic `NatsKvBucket` feature.** Every bucket in
  the platform opens through `NatsKvBucket.open`: `ReplayGuard`
  (`core/coordination/replay_guard.py:109-121`), `RevocationGuard` (`:267`), the idempotency
  store (`core/coordination/idempotency.py:373-379`), `KVLease`, `TokenBucket`,
  `BaseCollection`'s L2, `AgentConfigKV`. A stamp in `open()` would add a round trip per
  open under `client.py:2198`'s lock, consume revision 1 of every bucket, and insert a
  reserved key into buckets whose owners reason about emptiness. The *reopen signal* from
  half 1 is generic and free; the identity key is not.
- **The epoch bucket takes `ttl=None`.** `NatsKvBucket`'s `ttl` is per-bucket and becomes
  the stream's `max_age` (`kv.py:161`), so it would expire the identity key too, and the
  next opener would mint a fresh UUID. That turns a reset into a fleet-wide cache flush on
  a timer.
- **A leading underscore is legal but unprecedented.** nats-server's key grammar is
  `^[-/_=\.a-zA-Z0-9]+$` (mirrored as `_KV_KEY_GRAMMAR` at `collections/base.py:53`), and no
  key in this repo uses one. It is collision-safe because every `Subjects` builder prefixes
  the namespace, so no `Subject.path` can begin with `_`. The integration test asserts the
  round trip rather than assuming it.
- **Any new method on `NatsKvBucket` lands on `KvBucketLike` (`kv.py:437-464`)**, which
  forces `FakeKvBucket` (`core/testing/kv.py:59`) to mirror it under
  `tests/enforcement/test_fake_protocol_parity.py`. `FakeKvBucket` has no wipe/recreate
  method today; adding one is in scope for this task.
- **Public API growth requires a minor bump** under
  `tests/enforcement/test_api_growth_requires_a_minor_bump.py`, and CLAUDE.md's lockstep
  rule moves all ~27 packages together. Coordinate with epoch-task-01's bump rather than
  taking two.

## Naming

Use `uuid_utils.uuid7()`, not `uuid4`: `tests/enforcement/test_uuidv7_enforcement.py` hard-fails
any `uuid4` under `packages/*/src`. Stringify with an inline `# convert at border: <reason>`
marker, required by `tests/enforcement/test_uuid_stringification.py`.

The precedent to copy is `CollectionRegistry._origin_id`
(`core/collections/registry.py:111`): `str(uuid7())`, commented "An opaque token, never used
as a UUID", compared for equality only. That is exactly this discipline, already shipped.

Call the concept **identity**, consistently, and rename this file to match. Do not call it
a generation: `identity_generation` already exists in this repo as a single-writer fencing
generation (`core/security/identity_token.py:127`) and means something else.

## Why no seeding

Do not try to make the new counter start above the old one. Seeding from a clock cannot
guarantee "higher than last time" without remembering what last time was, which is the
durability this design removes. An earlier draft had surviving pods ratchet the counter
forward by purging the stream; `NatsKvBucket` exposes no purge, so that would have needed
new API on top of being wrong.

Identity comparison sidesteps it: the check is equality against the listener's own previous
reading, never an ordering comparison between machines. A changed identity says the old
numbers are meaningless, which is more useful than knowing they were merely lower.

## Acceptance

- A test wipes a real bucket via `js.delete_stream` (the one-liner at
  `test_kv_self_heal_round_trip.py:36`, against the `nats_container` fixture at
  `core/testing/fixtures.py:169`, already available to epoch per its
  `tests/integration/conftest.py`) and proves a listener holding a high `_last_seen`
  recovers rather than dropping bumps forever.
- A listener with two registered subjects fires `on_reset` for **both** when either detects
  the reset.
- After a reset, `_last_seen` is empty, not re-primed.
- A create race resolves to a single identity, with losers adopting the winner's.
- An unreadable identity is NOT treated as a reset, and leaves the recorded one alone.
- The listener issues exactly one kind of KV write: the create-if-absent that mints the
  identity, which IS the mechanism. It performs no other. (Stated this way because it is
  checkable;
  "the identity is opaque and never ordered" is not assertable by a behavioural test. If
  that guarantee needs enforcing, it is a static check under `tests/enforcement/`, not a
  unit test that cannot fail.)
