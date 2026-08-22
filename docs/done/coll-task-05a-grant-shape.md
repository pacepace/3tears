# coll-task-05a: Narrow the Minted Grant (3tears)

## Objective

`coll-task-03` puts a scope segment in the key; `-04` makes reads
subject-addressable. This shard narrows what the **auth callout** mints, and
closes the JetStream paths that route around a `$KV.` grant.

`coll-task-05b` does the static-user half in the hub. Both are required: only
`TOOL_POD` and `AGENT_POD` are ever minted, so this shard alone leaves most
principals untouched.

Facts cited here live in the evidence ledger.

---

## Why pinning the stream name pins nothing here

`mint_user_jwt` builds `kv_data` as `$KV.{bucket}.>` for every entry in
`permissions.kv_buckets`, and `_js_api_grants_for_stream` adds seven subjects per
stream. Its docstring argues that pinning the stream name denies the cross-stream
direct-read and destroy a bare `$JS.API.>` would allow -- true *between* streams,
but it assumes one stream per principal. `{ns}-collections` is one stream shared
by four principals today and more after `-07c`.

---

## The four bypasses this shard closes

**BYPASS 1 -- body-carried read.** `$JS.API.STREAM.MSG.*.{stream}` covers
`STREAM.MSG.GET`, whose key rides in the request body. Closed by `-04` plus
dropping this grant.

**BYPASS 2 -- bare direct get.** `$JS.API.DIRECT.GET.{stream}` without the `.>`
tail is get-by-sequence, also body-carried. Safe to drop: `NatsKvBucket` has no
read-by-revision call site.

**BYPASS 3 -- consumer create.** `add_consumer` serializes `filter_subject` and
`deliver_subject` into the request **body**, so a principal can create a consumer
filtered on `$KV.{bucket}.>` delivering to its own inbox. Both subject branches
(with and without a consumer name) are matched by the granted wildcards, so the
bypass does not depend on the bare form.

Nothing watches collections, so no capability is lost -- **except one**. Three
call sites reach `keys()`, which creates a watcher via `watchall()`:
`registry/catalog.py` and `agent_router/catalog.py` are on other buckets, but
`hub/admin/backup_engine.py`'s `_flush_nats_kv` enumerates
`js.key_value_store_names()` and therefore watches **and per-key deletes** every
bucket, collections included. GRANT-07 and GRANT-06 must carve out the hub's
restore path, or that path moves off `keys()`/`delete()`.

Note the blast radius is wider than one bucket: the `try:` opens **before**
`key_value_store_names()` and the `except Exception: log.warning` closes
**after** the whole nested loop, so one refused bucket silently aborts every
remaining bucket mid-flush. A restore would then leave other principals' scoped
copies behind, at WARNING.

**BYPASS 4 -- whole-stream export.** `$JS.API.STREAM.*.{stream}` puts the **verb at
token 4**. `SNAPSHOT` streams the entire bucket to a caller-named
`deliver_subject`; `RESTORE` is its write twin; `UPDATE` is also a read primitive,
since `republish`/`sources` mirror every key to a subject the caller controls.

---

## `$KV.` is publish-only, on every bucket

`mint_user_jwt` puts `kv_data` into **both** allow lists. Nothing in nats-py ever
subscribes `$KV.` -- `put` is a publish, `watch` subscribes an inbox, `get` is a
request. So a `$KV.` subscribe grant confers **no read capability** and hands the
holder a firehose of every write's full value.

Drop `kv_data` from the `sub` list **unconditionally**, not just for collections.
It costs nothing and closes the firehose on `checkpoints` and
`{ns}_agent_config` too -- the two cross-agent, cross-customer buckets the
sequence records as out of scope for *key* isolation.

The per-bucket opt-in below applies to the **publish** narrowing only.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| GRANT-01 | `kv_data` is removed from the subscribe allow-list for **every** bucket | P0 |
| GRANT-02 | The collections `$KV.` publish grant is emitted per-principal as `{scope}.>` -- never bare `>` | P0 |
| GRANT-03 | Publish scoping is **per-bucket opt-in**; every other bucket keeps `>` | P0 |
| GRANT-04 | No principal holds `$JS.API.STREAM.MSG.*` on the collections stream | P0 |
| GRANT-05 | On the collections stream, `$JS.API.STREAM.*` becomes literal verb tokens: `INFO` for every principal, plus `CREATE` and `UPDATE` for the declaring identity alone | P0 |
| GRANT-06 | No principal holds `SNAPSHOT`, `RESTORE`, `PURGE` or `DELETE` on the collections stream, save the carved-out restore path | P0 |
| GRANT-07 | No consumer verb is granted on the collections stream, save the carved-out restore path | P0 |
| GRANT-08 | No principal holds the bare `$JS.API.DIRECT.GET.{stream}` form | P0 |
| GRANT-09 | `$JS.API.DIRECT.GET.{stream}.>` is narrowed to the principal's scope | P0 |
| GRANT-10 | A scoped bucket whose principal has no scope is a mint-time error | P0 |
| GRANT-11 | `Principal` gains `AGENT_ROUTER` and `DATASET_EXECUTOR`, and the four currently-dead members are adopted rather than left unreferenced | P0 |
| GRANT-12 | `_js_api_grants_for_stream` is promoted to public `js_api_grants_for_stream` -- `__all__`, Sphinx docstring, exported from `threetears.nats` | P0 |
| GRANT-13 | Every principal that subscribes via `subscribe_typed` holds publish on `{ns}.deadletter.>` | P0 |

GRANT-05's carve-out is not a softening: `-04` makes hub bootstrap the canonical
declarer, which needs `CREATE` and `UPDATE`. Note the wildcard is **not** the
obstacle -- emitting three literal verb tokens expresses this exactly. The
obstacle is that `declare` is not least-privilege: `UPDATE` on a shared stream is
a read-all primitive (BYPASS 4). Bind it to the declaring identity alone and add
an enforcement rule that no pod principal may ever hold it.

**The literal narrowed read subject is
`$JS.API.DIRECT.GET.KV_{b}.$KV.{b}.{scope}.>`** -- `$KV` and `{b}` are separate
tokens. Writing `$JS.API.DIRECT.GET.{stream}.{scope}.>` matches nothing and
degrades to swallowed 10 s timeouts.

---

## Per-bucket opt-in is not optional

`kv_data` loops over every entry in `kv_buckets`. Agent pods hold `{ns}-epochs`,
`{ns}_agent_config`, `checkpoints`, `{ns}-ratelimits`,
`{ns}-proxy_assertion_nonces`; the hub holds more. **None writes a scope
prefix**, and `checkpoints` has its own separate `l2_key`. Emitting `{scope}.>`
for all of them denies every read on all of them -- as a broker timeout, not a
raise.

So `PrincipalPermissions.kv_buckets` stops being `tuple[str, ...]` and carries,
per bucket: name, optional scope, write intent.

`PrincipalPermissions.streams` is the same "resource + capability" concept in a
second shape, and the capability argument below serves both. Use one record for
both, or state why streams stay flat.

---

## Capability, not a blanket narrowing

Give `_js_api_grants_for_stream` a capability argument so each call site declares
what it needs on that stream: read-only KV, KV watch, durable consumer,
**declare**, admin. `declare` must be distinct from `admin`.

One live conflict to preserve: the registry declares the **result stream** at
startup through the `$JS.API.STREAM.*` wildcard. Different stream, keeps its
capability.

---

## GRANT-11: the missing and dead principals

`Principal` has exactly six members. Two processes that run L2 collections have
none -- `agent_router`, which owns `PodAffinityCollection` (sticky
conversation-to-pod routing), and `dataset_executor`. And four of the six that
exist -- `HUB`, `REGISTRY`, `GATEWAY`, `CHANNEL_ADAPTER` -- are referenced nowhere
outside `subject_permissions.py`, because those processes connect as static
users.

So `kv_key_scope_for` has no domain value for two confirmed consumers, and the
resolvers for four more are dead code.

**Add the two, adopt the four.** This shard already owns
`subject_permissions.py`, so it is the cheapest home, and both `coll-task-05b`
and `coll-task-06b` block until it lands -- `-06b` cannot wire a scope for a
principal that has no enum value.

`build-plan-principal-convergence.md` Chunk 11 also claims this work. It sits
behind ten unbuilt chunks, one of which is currently broken on a live branch, so
taking it here is the unblocking move; annotate Chunk 11 accordingly.

Note the adoption is a **grant-surface** change, not a migration: these
processes keep their static credentials until `-05b` moves them. What the enum
buys immediately is a legal, pinnable scope value.

---

## GRANT-12: promote the grant builder so the conf side can generate from it

`coll-task-05b` must deny each static user's `$JS.` reach into the collections
stream, and the only correct source for those subjects is the function that
emits them. Hand-deriving the shapes has now failed twice -- once on whole-token
wildcards, once by missing `$JS.API.STREAM.MSG.*.{stream}` (six tokens,
terminal, which is BYPASS 1).

`-05b` lives in the hub, so importing `_js_api_grants_for_stream` across a repo
boundary would be a Shape-A underscore violation. Per promote-not-exempt, the
fix belongs here: make it public, and have `-05b` call it.

---

## GRANT-13: the deadletter grant nobody holds

`subscribe_typed` routes validation failures to `{ns}.deadletter.{subject}`.
Grepping `subject_permissions.py` for `deadletter` returns **nothing** -- no
principal is granted it. The registry is incidentally covered by its static
`aibots.>`; a callout-minted agent pod is not, so its deadletter publish is
refused.

This is pre-existing, and `coll-task-07a` makes it load-bearing by moving two
more consumers onto `subscribe_typed`. Fix it here, where the resolvers live.

---

## The registry needs the grant it does not have

`_registry`'s `kv_buckets` is `("tool_catalog", f"{ns}-pop_nonces")` -- no
collections bucket. But `registry/server.py` configures `l2_client=nc` and builds
`HeartbeatCollection`, which is **`L3 = None`**. So the registry runs a
source-of-truth collection against a bucket it is not granted, working today only
because the static `registry` user carries `$KV.>`. That is a data-loss case, not
a cache miss.

Add `{ns}-collections` to `_registry`. Note this only takes effect once
`coll-task-05b` moves that principal off its static grant, or Chunk 11 adopts it
onto the callout -- see the ledger's Part 5.

---

## On the dead grants -- record, do not delete

Seven granted bucket names have no bucket on dev. That is evidence, not a
verdict; the file carries a standing comment from the last time someone read "no
opener found" as "no opener", and another warning that normalising these names
renames a live bucket. Two are known live (`{ns}-leases`, `{ns}-epochs`). Not
this shard's work.

---

## Files to Modify

- `packages/nats/src/threetears/nats/subject_permissions.py` -- the per-bucket grant record; scope and write intent in the resolvers; `{ns}-collections` on `_registry`.
- `packages/nats/src/threetears/nats/user_jwt.py` -- `kv_data` construction, its removal from the sub list, the capability argument.
- `packages/nats/src/threetears/nats/_diagnostics.py` -- `kv_grant_remedy` currently tells operators *"`mint_user_jwt` expands one entry into pub+sub on `$KV.{bucket}.>` … add it there too."* Post-landing that is actionable-wrong: it instructs reopening the hole.
- `3tears/tests/enforcement/test_kv_bucket_grant_naming.py` -- repo root, not `packages/nats/`. Three targeted pinned-pair tests (KVLease, registry catalog, epoch bucket).
- `packages/nats/tests/unit/test_user_jwt.py`, `.../test_subject_permissions.py` -- both construct or assert string membership against `kv_buckets` as a plain tuple.
- `packages/nats/tests/integration/test_user_jwt_scoped_grant_live.py` -- same, plus the probes.

---

## Anti-patterns

- DO NOT narrow `$KV.` publish uniformly across buckets. Every non-collections bucket dies as a broker timeout.
- DO NOT keep `$KV.` on subscribe anywhere. It confers no read.
- DO NOT enumerate destructive verbs to deny. Allow-list literal verbs.
- DO NOT drop consumer verbs without the backup-restore carve-out.
- DO NOT grant `declare` to a pod principal. `UPDATE` is a read-all primitive on a shared stream.
- DO NOT delete a grant because the bucket is absent on dev.
- DO NOT use the word "tenant" -- banned, and here also wrong: the point is that principals *share* a stream.

---

## Success criteria

- [x] `$KV.` appears in no subscribe allow-list -- `mint_user_jwt`'s `sub.allow` is now `permissions.subscribe` alone
- [x] The collections publish grant is `{scope}.>`; every other bucket unchanged at `>` -- per-resource opt-in via `JsResource.scope`
- [x] The collections stream's JS grants are literal `INFO` + the scoped `DIRECT.GET` tail, plus `CREATE`/`UPDATE` for the declaring identity alone (`_hub`, pinned by `tests/enforcement/test_kv_grant_capability.py`)
- [ ] **The backup-restore path still works -- NOT met here, and it cannot be.** The hub reaches `kv.keys()` (a JetStream consumer) and per-key `kv.delete()` on every bucket through its static `>`, which this shard does not touch and `-05b` records as a dated residual. Neither operation is expressible under a scoped capability, so the carve-out has to be a hub-side change: see "What the hub must do", recorded in `_hub`'s own resolver comment so `-05b` cannot miss it
- [x] `_diagnostics.kv_grant_remedy` no longer advises the old shape -- pinned by `test_the_remedy_does_not_instruct_the_reader_to_reopen_the_hole`
- [x] `test_kv_bucket_grant_naming.py` and the three test modules updated and green
- [x] `./scripts/check-all.sh` (15897 passed, 3 skipped, 411 deselected; 139 sidecar) and `./scripts/test-integration.sh` (392 passed, 19 skipped) green

### What the hub must do (this shard may not edit it)

`hub/admin/backup_engine.py`'s `_flush_nats_kv` enumerates `js.key_value_store_names()`,
then `kv.keys()` (→ `watchall()` → a JetStream consumer) and per-key `kv.delete()` on
every bucket, collections included. Three separate problems, in order of blast radius:

1. **The `try:` opens before `key_value_store_names()` and the `except Exception:
   log.warning` closes after the whole nested loop**, so ONE refused bucket silently
   aborts every REMAINING bucket mid-flush. A restore then leaves other principals'
   scoped copies behind, at WARNING. Per-bucket error handling first, regardless of
   grants.
2. **Consumer-create is not grantable under a scoped capability** -- it is BYPASS 3,
   the create-with-`filter_subject` read. Move off `keys()`: the bucket is memory
   storage with no durable data, so deleting and re-declaring it is both simpler and
   what a fresh cluster does anyway.
3. **A per-key `delete` outside the hub's own scope is not grantable either** -- it is
   a `$KV.{bucket}.{other_scope}.…` publish. Same fix as (2).

Until then the path keeps working on the hub's `>`, which is why this is a recorded
residual rather than a break.

---

## Verification

```bash
cd 3tears
./scripts/check-all.sh
./scripts/test-integration.sh
```

Extend `packages/nats/tests/integration/test_user_jwt_scoped_grant_live.py`.
From principal A, each **refused**:

```
$JS.API.DIRECT.GET.KV_<b>.$KV.<b>.<other_scope>.widgets.e1
$JS.API.STREAM.MSG.GET.KV_<b>  {"last_by_subj":"$KV.<b>.<other_scope>.widgets.e1"}
$JS.API.CONSUMER.CREATE.KV_<b> {"stream_name":"KV_<b>","config":{"filter_subject":"$KV.<b>.>","deliver_subject":"<A inbox>"}}
$JS.API.STREAM.PURGE.KV_<b>
$JS.API.STREAM.SNAPSHOT.KV_<b>
$JS.API.STREAM.UPDATE.KV_<b>
$KV.<b>.<other_scope>.widgets.e1
```

Each **succeeding**: A's own scope, read and write.

These are `pytest.mark.integration` and `test.sh` runs `-m "not integration"`.
**Make the run, not the collection, the success criterion.**

---

## What landed differently from the shard

- **One record for BOTH kinds, and one field.** `kv_buckets` and `streams` are gone;
  `PrincipalPermissions.js_resources` is a single tuple of `JsResource`, each carrying
  name, kind, capability, scope and write intent. The shard offered "or state why
  streams stay flat" -- they did not.
- **Three capabilities, not five.** `FULL` (the historical seven subjects, unchanged),
  `KV_SCOPED`, `KV_SCOPED_DECLARE`. Splitting `FULL` further into "KV watch" and
  "durable consumer" would be better least-privilege but is not supportable on
  evidence anyone has gathered: every non-scoped bucket runs `allow_direct: false`,
  where the read is `STREAM.MSG.GET` with the key in the body, so trimming verbs there
  would deny reads that work today -- as a deadline, not an error. `FULL` is therefore
  the historical set verbatim and the narrowing is confined to the resources that
  actually write a scope. Recorded in `JsCapability`'s own docstring.
- **A pod principal whose id is not a UUID can no longer be granted at all.** GRANT-10
  lands at the RESOLVER: `_agent_pod` derives its scope through `kv_key_scope_for`,
  which refuses a non-uuid, so `build_permissions(AGENT_POD, agent_id="agent-A", …)`
  now raises. That is the fail-closed direction and it is the point, but it is a
  behaviour change with a test-surface cost: every pod id in
  `test_subject_permissions.py` and `test_kv_bucket_grant_naming.py` is now a UUID.
  A tool pod is unaffected until `coll-task-07c` gives it the bucket.
- **A cost `-06x` must absorb.** Pods still default `create_if_missing=True`, so a pod
  opening the collections bucket now issues a `STREAM.CREATE` it is refused. The
  refusal is not an error -- `open_kv_stream` waits out the JetStream deadline and then
  falls through to the bind, which succeeds. So it works, once, slowly. KVC-04's
  `configure(l2_create_if_missing=False)` is what removes the stall, and `-06x` owns it.

## Enforcement test suggestions

Build on `threetears.enforcement.common`, which already exports `Violation`,
`iter_python_files`, `parse_python_file`, `parse_exemptions_with_rationale`,
`apply_exemptions`, `emit_report` and `resolve_mode`, and follow the established
mode-env-var plus `_*_exemptions.txt`-with-rationale shape. `tests/enforcement/test_no_bespoke_reuse.py`
is the worked example, and its own docstring notes it delegates to the shared
walker rather than re-rolling one -- re-rolling here would break the rule this set
is trying to enforce.

Landed as `tests/enforcement/test_kv_grant_capability.py` (`KV_GRANT_ENFORCEMENT_MODE`,
default `strict`; exemptions in `_kv_grant_exemptions.txt`), three AST walkers on
`threetears.enforcement.common`, each with a planted-violation self-test AND a
stays-quiet-on-sanctioned-source test:

- [x] **No pod principal holds the `declare` capability** -- `find_declare_outside_the_declarer`, both the `declare=True` factory shape and the raw `JsCapability.KV_SCOPED_DECLARE` constructor shape, sanctioned set `{_hub}`. Plus a non-vacuity test that the hub genuinely still holds it, so the rule cannot pass by the declaration having been deleted.
- [x] **No verb wildcard survives on a scoped stream** -- asserted behaviourally in `test_no_verb_wildcard_survives_on_a_scoped_stream` rather than by a string walker, because the property is about the EMITTED grants and a walker over literals would miss the composition.
- [x] **Every bucket declares scope and write intent explicitly** -- `find_implicit_bucket_decisions`, backed by `JsResource.kv`'s keyword-only, default-less parameters.
- [x] **The shared bucket is never granted unscoped** -- `find_unscoped_shared_bucket_grants`, the regression an added principal actually produces.
