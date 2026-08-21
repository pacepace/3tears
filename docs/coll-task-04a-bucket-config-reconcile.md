# coll-task-04a: Bucket Config Reconcile (`allow_direct`)

## Objective

Make `coll-task-05a`'s narrowed grant enforceable on reads, by running the
collections KV bucket with `allow_direct: true`.

Getting there means giving KV buckets a create-or-reconcile primitive and
deciding who owns canonical bucket config. Facts cited here live in the evidence
ledger.

---

## Why this blocks the grant

The live bucket reports `allow_direct: false`.

nats-py takes the flag from the **server's** stream info — both `key_value()`
and `create_key_value()` set it from `bool(si.config.allow_direct)`, never from
the requested config — stores it on the `KeyValue` instance, and branches on it
in both `_get` read paths. With it false, a `get` is:

```
subject: $JS.API.STREAM.MSG.GET.KV_aibots-collections
body:    {"last_by_subj": "$KV.aibots-collections.<key>"}
```

**The key is in the body.** NATS matches subjects, so a key-scoped `$KV.` grant
constrains nothing on reads. With it true, nats-py issues
`$JS.API.DIRECT.GET.{stream}.{subject}` — the key is the subject tail, and it is
pinnable.

That chain verifies cleanly against installed nats-py and is the load-bearing
claim of the sequence.

---

## The defect: create-or-bind discards the requested config

`NatsKvBucket.open` tries `create_key_value(...)` and, on exception, binds to the
existing bucket in an `except` branch that **drops the caller's config**. (If the
bind itself also fails it raises `KvError` with the grant remedy — that arm is
fine; the silent one is the successful bind.)
There is a `log.debug` for the bind; nothing at WARNING or above says the
requested config was not applied.

`NatsKvBucket.open` is the one JetStream opener in 3tears that does not follow
the existing pattern. `NatsClient.ensure_jetstream_stream` is already the
create-or-reconcile primitive for streams.

**One correction, because the anti-pattern below depends on it.**
`ensure_jetstream_stream`'s classification is **two-way, not three-way**: it
types only subjects-overlap (matched on err_code 10065 *or* the substring
"subjects overlap") and falls through to `update_stream` for *every* other add
failure — "already in use" and a **permission refusal alike**. So it does not
already show how to distinguish "refused". That arm has to be built.

### The config is not "already observably" dropped

The live values are what the code asks for: `max_age` unlimited because
`kv_bucket` defaults `ttl=None`; memory storage by default; `allow_direct`
**omitted entirely** from the create body (`KeyValueConfig.direct` defaults to
`None` and `Base.as_dict` skips None fields — it is not sent as `null`);
`discard` and `deny_delete` hardcoded by nats-py.

The 7200 s TTL belongs to the retired `NatsKvClient`, which has zero production
construction sites. **The defect is real in source and has now been demonstrated
live** (2026-08-20, testcontainers NATS): opener A created a bucket with
`ttl=60s, history=1`; opener B, on a fresh client with an empty bucket cache,
asked the same bucket for `ttl=7200s, history=5`. The server still reported
`max_age: 60.0, max_msgs_per_subject: 1`, while `NatsKvBucket.ttl` on B's handle
reported `2:00:00` — the wrapper reports back the value the server never applied.
The only trace was `DEBUG ... KV bucket bound (already existed)`. Do not cite the
`NatsKvClient` TTL; cite this.

That class does carry one genuine contradiction: its docstrings claim **file**
storage while `BucketConfig.storage` defaults to memory. Its TTL half is
self-consistent; only storage is stale.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| KVC-01 | A KV create-or-reconcile primitive exists on `NatsClient`, building the KV `StreamConfig` itself | P0 |
| KVC-02 | Opening an existing bucket whose compared config differs raises `KvConfigMismatch` — **not** a `KvError` subclass | P0 |
| KVC-03 | The collections bucket is declared canonically by hub bootstrap, beside the existing stream declarations | P0 |
| KVC-04 | `create_if_missing` **stays `True` by default**, and is a `configure()`-level flag rather than a literal in `base.py` — not set to `False` until KVC-10 lands | P0 |
| KVC-05 | Each process opens the collections bucket eagerly at startup, so a mismatch raises at wiring rather than in a request path | P0 |
| KVC-06 | The live bucket ends the landing with `allow_direct: true` | P0 |
| KVC-07 | Reconcile compares a named field set — `direct` for this landing | P0 |
| KVC-08 | ~~`deny_purge: true` on the collections stream~~ **WITHDRAWN — impossible** (see below) | P1 |
| KVC-09 | The retired `NatsKvClient` is deleted — **not this shard**; owned by `coll-task-04b-delete-nats-kv-client.md` | P1 |
| KVC-10 | The collections bucket survives a NATS restart without a hub restart, via a hub-side `ensure_kv_bucket` on NATS **reconnect** | P0 |

**KVC-10 takes the reconnect branch, not `storage: file`.** A `file`-storage
branch was considered and rejected: CLAUDE.md's Config Source-of-Truth carve-out
ratifies that the `BaseCollection` L2 cache stays `storage="memory"` and that the
named volume deliberately does not push it to disk. (That carve-out cites
`NatsKvClient.storage` as its authority — the class KVC-09 deletes, whose
docstrings claim *file* storage against a memory default. The real authority is
`NatsClient.kv_bucket`'s `storage: str = "memory"`; say so in the CLAUDE.md edit
this shard already schedules.) Reversing that is a bigger
decision than this shard, and it composes badly with unlimited `max_age` plus
`coll-task-03`'s tombstones. The reconnect hook already exists —
`NatsClient.add_reconnect_callback`, dispatched from the `_dispatch_reconnected`
closure installed as nats-py's `reconnected_cb`. (`_on_reconnected` is log-only —
it is called *by* that closure, not the dispatcher.) The one production caller to
copy is `aibots_agents/runtime/bootstrap/orchestrator.py`'s
`_on_nats_reconnected`.

---

## The exception type is the whole trick

`KvError` is what `BaseCollection`'s L2 accessors catch and degrade on, with
`_ensure_kv` deliberately inside the catch. Raising `KvError` on a config
mismatch would be **downgraded to a per-op WARNING and the fleet would run with
L2 silently disabled** — the exact degradation `coll-task-03` rejects. A "fail
loud" the caller already swallows is not fail-loud.

Hence `KvConfigMismatch` as a distinct type the accessors do not catch.

**And it must be raised at startup, not lazily.** Pods resolve the bucket in
`_ensure_kv`, which runs on first read — so an un-caught mismatch would escape
into a request path under load, reproducing the failure mode `coll-task-03`
argues against and colliding with `coll-task-06x`'s "no process raises on first
L2 access".

**KVC-05's per-process half belongs to the `coll-task-06x` shards**, which edit
exactly the `configure()` call sites it needs to sit beside — one line each,
`await nc.ensure_kv_bucket(...)`. Landing it here would put it in one process and
miss the other ten. No new startup-probe abstraction is needed: the platform's
existing shape is a declaration call in lifespan or `start()`, with
`threetears.observe.health.HealthCheck` / `HealthTier` for the orchestrator's
LIVE-vs-READY question.

**Rollout ordering:** the hub must reconcile the live bucket to
`allow_direct: true` **before** pods roll out with an eager open, or every pod
crash-loops on `KvConfigMismatch`. Inside hub lifespan, the bucket declaration
must precede `registry.configure(l2_client=nc)`.

A second swallow sits on the same path: `NatsKvClient.connect` wraps the open in
`except Exception: log.warning(...)`, commented "fail-open". KVC-09 removes that
class outright. If deletion slips, narrow the catch instead.

---

## Reconcile policy

`ensure_jetstream_stream` updates in place, and that is the right consistency
for both JetStream resource kinds. `coll-task-05a` removes `STREAM.UPDATE` from
pod principals, so a pod cannot update even if the primitive can. Those
reconcile:

- **The primitive can update.**
- **Only the declaring identity is granted to.** The hub declares the collections
  bucket in lifespan beside the streams it already declares there, with the same
  "idempotent: binds to the existing stream on restart" reasoning.
  `coll-task-05a`'s GRANT-05 must carry a matching `declare` capability for that
  identity — an INFO-only allow-list with no exception makes this requirement
  unexecutable and a cold cluster unable to bootstrap.
- **Pods pass `create_if_missing=False`** for the collections open specifically —
  via the `configure()`-level flag of KVC-04, and only once KVC-10 has landed.

This satisfies the Config Source-of-Truth rule — pods are readers.

### `create_if_missing` stays `True` globally

Flipping the `kv_bucket` default would break every bucket that relies on
first-use creation — the coordination primitives, epoch client, IAM stores,
memory extraction, three hub caches — all of which the scope note below
enumerates and enumerated in the scope note below.

It would also neuter `NatsKvBucket._reopen`, which re-runs `open()` with the
stored flag and **exists because a single-node NATS restart on ephemeral
JetStream storage wiped every bucket and silenced the wake scheduler in
production**. The collections bucket is memory storage with one replica: it does
not survive a restart, and if pods cannot recreate it and only the hub declares
it at lifespan, L2 is off fleet-wide, at WARNING, until the hub restarts.

So: default unchanged; readers-not-writers is enforced by the **grant**, where a
denied create is fail-closed by construction.

**And `create_if_missing=False` must not be hard-coded in `base.py`.** That would
bake one deployment's bucket-ownership policy into the library, and it would
neuter `_reopen` for the collections bucket specifically — the one place it is
fatal, since that bucket is memory-storage with one replica and the hub declares
it only in lifespan. A NATS restart would then leave L2 off fleet-wide, at
WARNING, until the hub restarts.

Make it a `configure()`-level flag the wiring sets, and **do not set it to
`False` until KVC-10 lands.**

Note the platform already has a precedent for this tension: the hub declares its
three streams in lifespan *and* each consumer ensures the identical stream in its
own `start()` — `hub/app.py`'s comment on the agent-router stream says
"(declaration cannot drift)". Dual declaration, not single ownership. KVC-05's
eager per-process open is the same shape.

**One more reason the grant cannot carry this alone:** "readers-not-writers is
enforced by the grant" holds only for the two callout-minted principals. Every
static user has `$KV.>` and `$JS.>` until `coll-task-05b` lands, so after a NATS
restart whichever static process opens first recreates the bucket with `direct`
unset — and every pod doing KVC-05's eager open then crash-loops on
`KvConfigMismatch`. Add a criterion that no static-user process can create the
collections bucket, and sequence accordingly.

---

## Building the KV stream shape

`create_key_value` never sets `deny_purge`, so KVC-08 is unreachable through it.
`ensure_kv_bucket` must go through `add_stream`/`update_stream` and **reproduce
the whole KV stream shape** — nineteen fields, including `deny_delete`,
`discard=NEW`, `allow_rollup_hdrs`, `max_msgs_per_subject=history`,
`duplicate_window`, `allow_msg_ttl`, `subject_delete_marker_ttl`, `max_msgs=-1`,
`max_consumers=-1` and `subjects=["$KV.{b}.>"]`.

**Read `create_key_value` and mirror it rather than working from a list** — any
enumeration here will go stale, and the failure is quiet: miss a field and the
"bucket" stops behaving as one, with nats-py's `key_value()` validating
`max_msgs_per_subject >= 1` and raising `BadBucketError`.

Landed as `build_kv_stream_config` in `packages/nats/.../kv.py`, guarded against
staleness by `test_kv_stream_shape_matches_nats_py` — it builds one bucket each
way against a live broker and compares the two server-side stream configs, so a
nats-py change fails a test instead of a deployment.

### KVC-08 is withdrawn: `deny_purge` is unreachable on a KV stream

`nats-server` refuses `allow_rollup_hdrs` and `deny_purge` together —
`roll-ups require the purge permission`, err_code **10052** — and
`allow_rollup_hdrs` is part of what MAKES a stream a KV bucket
(`create_key_value` hardcodes it). Turning it off to win `deny_purge` produces a
stream that accepts writes and then fails `KeyValue.purge` with
`rollup not permitted` (err_code 10111): the exact "stops behaving as a bucket"
failure the paragraph above warns about.

So the requirement cannot be met through stream config at all, and the shard's
premise for it — "`create_key_value` never sets `deny_purge`, so KVC-08 is
unreachable through it" — was right about the method and wrong about the
conclusion: it is unreachable through anything.

Protecting the shared bucket from `$JS.API.STREAM.PURGE` therefore stays where
the rest of that surface already is: `coll-task-05a`'s grant narrowing (bug #10,
"`$JS.API.STREAM.*` covers SNAPSHOT/RESTORE/PURGE/UPDATE on a shared stream").

Proven live and pinned so the verdict is re-checked on every nats-server the
suite runs against, rather than believed on one reading:
`test_deny_purge_is_not_expressible_on_a_kv_stream`.

Note also that `NatsClient.kv_bucket` caches by full name, so the first opener's
config wins for the process lifetime. `ensure_kv_bucket` must share that cache
or hub bootstrap and hub collections will diverge on `direct`.

KVC-07 compares a **named field set**, not everything: requested and server
config differ on some field on essentially every open, so a full comparison
would raise forever. For this landing the set is `direct`. Consequence worth
stating: `deny_purge` is set at create and never reconciled after.

---

## Flipping the live bucket

Memory storage, no durable data, read-through. Delete the bucket and let
bootstrap recreate it. No migration, no dual-read, no shim. Do not rely on
entries ageing out — `max_age` is unlimited.

`allow_direct` can also be flipped in place with `update_stream`, worth knowing
for production, but delete-and-recreate exercises the path a fresh cluster needs
anyway.

**Precondition to record:** `allow_direct` permits follower/mirror reads. That is
safe at `num_replicas: 1` with `mirror_direct: false`, which is the live config
— but this bucket hosts `l2_cas_mutate` users where L2 is the source of truth,
and a stale direct read breaks CAS. Raising replicas requires re-examining every
such collection in the bucket.

---

## Files to Modify

- `packages/nats/src/threetears/nats/client.py` — `ensure_kv_bucket`; thread `direct` through `kv_bucket` (no such parameter today); share the bucket cache.
- `packages/nats/src/threetears/nats/kv.py` — `NatsKvBucket.open`'s **create** branch routes through `ensure_kv_bucket` instead of calling `create_key_value` directly, and the bind branch delegates or refuses. The create branch matters most: `_reopen` re-runs `open()` after a NATS restart, so leaving it on `create_key_value` means the hub's own cached handle silently recreates the bucket with `direct` unset, racing KVC-10.
- `packages/nats/src/threetears/nats/errors.py` — `KvConfigMismatch`, subclassing `NatsClientError` beside `KvError` and `StreamSubjectsOverlapError`. Every nats exception has one home.
- `packages/core/src/threetears/core/collections/registry.py` — the `configure()`-level `create_if_missing` flag. **Not** a literal in `base.py`; `base.py`'s `_ensure_kv` reads the flag.
- `packages/core/src/threetears/core/cache/kv.py` — **delete** `NatsKvClient` (KVC-09); decide `BucketConfig`'s fate.
- `packages/core/tests/test_kv_client.py` — deleted with it (24 tests).
- `3tears/tests/enforcement/test_dict_state_detection.py` — catalog entry naming `NatsKvClient`.
- `14-eng-ai-bot-agents/tests/unit/runtime/conftest.py` — patches `threetears.core.cache.kv.NatsKvClient` by path; `mock.patch` raises on a deleted target, and this is a shared fixture.
- `14-eng-ai-bot/src/aibots/hub/app.py` — declare the collections bucket in lifespan.
- `14-eng-ai-bot/CLAUDE.md` — its cache carve-out cites `NatsKvClient.storage` as the authority for memory storage.

---

## Scope note: the defect is wider than collections

Every one of these opens a bucket through `kv_bucket` and inherits the discard:
the coordination primitives (`lease`, `idempotency`, `replay_guard`,
`token_bucket`, `windowed_counter`, `distributed_counter`),
`nats/distributed_lock.py`, `epoch/client.py`, `iam/stores/nats_kv.py`,
`agent/memory/extraction.py`, and three hub caches (`channel_resolve_cache.py`,
`principal_auth_cache.py`, `diagnostics/context_capture.py`).

Fixing the primitive fixes all of them. Scope only the *canonical declaration*
and the `direct` field to collections.

---

## Anti-patterns

- DO NOT raise `KvError` on a mismatch. It is caught and degraded.
- DO NOT raise the mismatch lazily. It lands in a request path under load.
- DO NOT flip `create_if_missing`'s default. It breaks ~12 buckets and neuters the restart self-heal.
- DO NOT add `direct=True` to `KeyValueConfig` and stop. On an existing bucket it changes nothing, and it will look done.
- DO NOT assume `ensure_jetstream_stream` already distinguishes a refusal. It does not; build that arm.
- DO NOT cite the 7200 s TTL as evidence of drift.
- AVOID `nats stream ls` for bucket state — it hides KV streams.

---

## Success criteria

- [x] `ensure_kv_bucket` exists, builds the full KV stream shape, and shares the bucket cache
- [x] Opening an existing bucket with a differing `direct` raises `KvConfigMismatch` — on the BIND path (`create_if_missing=False`), which is the reader's contract. The declaring path (`create_if_missing=True`) reconciles in place instead, per "the primitive can update"
- [x] That type is not caught by the L2 accessors, and is raised at startup
- [x] `js.key_value()` still binds after a reconcile
- [x] `create_if_missing`'s default is unchanged, and it is a `configure()` flag — no literal in `base.py`
- [x] KVC-10's reconnect hook is wired — `hub/app.py`'s `_redeclare_collections_bucket`, registered via `NatsClient.add_reconnect_callback`. Statically pinned by `tests/enforcement/test_collections_bucket_declaration.py`; the equivalent behaviour (JetStream wiped out from under a live handle, bucket returns carrying `allow_direct`) is proven live by `test_the_self_heal_recreates_with_direct`. **Not yet proven by killing NATS under a running hub** — the hub cannot boot until `coll-task-06b` supplies its `kv_key_scope`
- [ ] No static-user process can create the collections bucket — `coll-task-05b`
- [x] Hub lifespan declares the collections bucket, before `registry.configure(l2_client=nc)`
- [ ] `NatsKvClient` is gone; no import or patch target remains in any repo — `coll-task-04b`
- [ ] Live bucket reports `allow_direct: true` — the MECHANISM is proven against a live broker (`test_declaring_flips_allow_direct_on_a_live_bucket`: a bucket created by `create_key_value` with `allow_direct: false`, holding a value, ends the call with `allow_direct: true`, still bound, value intact). **Not yet run against cobalt-dev or the local stack**, because the hub cannot boot until `-06b`
- [x] `./scripts/check-all.sh` and `./scripts/test-integration.sh` green

---

## Verification

```bash
cd 3tears
./scripts/test.sh nats -v
./scripts/check-all.sh
./scripts/test-integration.sh
cd ../14-eng-ai-bot-agents
uv run pytest tests/unit/ -q
```

Live: `nats stream info KV_aibots-collections -j` — `allow_direct` true after.

Behavioural, and the one that matters: with `allow_direct: true` and
`coll-task-05a`'s grant in place, a principal reads its own key and is **refused**
on another's, via both the direct and the body-carried form. If the body-carried
form still succeeds, this shard did not land.

---

## Enforcement test suggestions

Both landed in `3tears/tests/enforcement/test_kv_bucket_open_discipline.py`
(`KV_OPEN_ENFORCEMENT_MODE`, default `strict`; exemptions in
`_kv_open_exemptions.txt`), each with a self-test proving its walker fires on a
planted violation and stays quiet on sanctioned source.

- [x] **No production code opens a KV bucket outside the reconcile primitive.** AST walker over `create_key_value` / `key_value` call sites outside `threetears/nats/kv.py`. It found one real site on its first run: `registry/server.py`'s tool catalog, exempted with a specific rationale (a BARE unprefixed bucket name `kv_bucket` cannot express, requesting no config, whose live storage type is not this shard's to change).
- [x] **`KvConfigMismatch` never appears in an `except` clause that also handles `KvError`.** The specific way this fix gets silently undone.

Plus, in the hub: `tests/enforcement/test_collections_bucket_declaration.py` pins
that lifespan declares the bucket with `direct=True`, that the declaration
precedes `registry.configure(l2_client=)` (the bucket cache makes the ordering
load-bearing), and that a registered reconnect callback re-declares it.
