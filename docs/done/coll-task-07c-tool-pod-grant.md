# coll-task-07c: Grant a Tool Pod Its Own L2 Tier

## Objective

Let a tool pod hold a `BaseCollection` with a working L1 and L2 tier, scoped so
no other principal can reach its data and it can reach no one else's.

Last in the sequence: this is the grant the previous shards make safe. Depends on
`coll-task-07a` (one consolidated stack to wire into).

L3 is `design-l3-for-non-agent-principals.md`. Two tiers is a complete pattern.

Facts cited here live in the evidence ledger.

---

## The gap

`_tool_pod` grants exactly two KV buckets -- `{ns}-proxy_assertion_nonces` and
`{ns}-leases`. **No `{ns}-collections`.**

`CROSS_PLATFORM_CACHE_INVALIDATE` appears in the pub and sub tuples of the agent
pod, registry, hub, gateway and channel adapter, **and in neither of the tool
pod's**. So a tool pod can neither announce a write nor learn of one.

The tool pod is one of two principals missing the collections grant; the registry
is the other, and `coll-task-05a` fixes it.

### What a missing grant does -- three cases

- **raw JS control-plane call on an ungranted subject**: does not raise. Blocks to
  the deadline and reads as an unreachable broker.
- **bucket open through `KVLease`**: **raises `KvError`** after a JetStream
  timeout, which nothing catches -- a hard failure on first claim.
- **bucket open through a `BaseCollection`** -- which is what this shard is about:
  the same `KvError` is **caught and degraded to a WARNING**, because
  `_get_from_l2`/`_save_to_l2`/`_delete_from_l2` deliberately put `_ensure_kv`
  inside the catch. Only `l2_cas_mutate` propagates. So an ungranted tool-pod
  collection presents as a silent warning and a dead cache, not a crash. That is
  the debugging trap, and an earlier draft of this section had it backwards.
- **over-narrowed `$KV.` data subject**: the publish is refused, nats-py routes
  the permission error to `_error_cb` without cancelling the request, and the op
  blocks the full 10 s KV timeout before surfacing as a swallowed warning.

Expect the right one for the path you are debugging.

---

## The scope: `tool_pods.id`

The pod's registry primary key, which is already its authenticated `claims.sub`.
Configured once per deployment, shared by every replica, hub-owned, present at
connect.

**This needs no new plumbing.** `_resolve_tool_pod` already pins it.

A namespace-derived scope was designed and rejected on evidence -- worth recording
because the reasoning generalizes:

- `tool_namespace_id` mints **one row per tool** -- see the ledger's Part 2 for
  the counts and their anchors. The shape is what matters, not the number.
- It is a pure function of manifest values **the pod sends**, bounded only by an
  `allowed_namespaces` prefix test -- so a pod could mint unbounded new scopes into
  a memory-backed bucket with no `max_bytes`.
- Its docstring says platform pods deliberately **collide** on one row, which is
  the opposite of the isolation property a scope must have.
- Rows are written by `tools.register`, **after** the JWT is minted, so no such
  value exists at connect.

An earlier draft of this shard carried the anti-pattern *"DO NOT scope by
`pod_id`. Every replica gets its own cache."* That premise was never verified and
is **false for tool pods**: `tool_pods.id` is per-deployment, not per-process.

### Multi-namespace pods: answered

Source answers it -- every platform tool pod serves many namespaces. An earlier
"if none does, refuse" branch would have refused the platform's own tool servers.
Scoping by `tool_pods.id` makes the question moot; the collection does not need
to know which namespace it belongs to.

---

## The invalidation grant is a decision, not a detail

Granting a partner-operated pod pub and sub on `threetears.cache.invalidate` -- a
single non-namespaced global subject carrying `{table, ids, origin}` with the raw
pk of every write -- gives it:

- a **cross-customer metadata firehose**: every table name and entity id written
  by every agent, hub replica, gateway and adapter;
- a **fleet-wide eviction primitive**: forged messages evict L1 everywhere, and
  with no L1 max age set anywhere that is an unbounded stampede onto YugabyteDB.
  `_on_invalidation` filters only on self-asserted `origin` -- trivially spoofed by
  asserting a different one -- and drops the **RBAC scan cache before** the
  collection, L1 and has-table guards, so a forged message naming any table drops
  a security cache on pods that hold no such collection.

**The decision, so an implementer does not have to make it:** grant the global
subject, record the exposure as accepted for this landing, and defer both
`origin` authentication and any move to a scoped subject.

The reasoning: the static NATS users already hold this subject unprefixed, so a
tool pod is not the widest holder and scoping it alone would buy little
(closing that side is `coll-task-05b`'s). Both
deferred items are publisher-side changes across three repos -- a wire-protocol
change, not a grant change -- and they need their own shard with their own
review. Carrying them as a P0 checkbox here would make the implementer the
ratifier of a security posture.

**`coll-task-08-invalidation-origin-auth`** owes two things: `origin` minted from
the authenticated principal, and `_on_invalidation` rejecting a table the
receiver does not hold **before** touching the scan cache. It is not written yet;
the name exists so the deferral is findable rather than floating. Note the guard reorder overlaps
`coll-task-03`'s L2S-09, which restructures the same function -- sequence them or
land them together.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| TP-01 | A tool pod is granted `{ns}-collections`, scoped to its `tool_pods.id`, on **publish only** (read is the scoped `DIRECT.GET` tail) | P0 |
| TP-02 | The tool pod is granted the **global** invalidation subject, with the exposure below recorded as accepted; `origin` authentication and any scoped-subject move are deferred to `coll-task-08-invalidation-origin-auth` | P0 |
| TP-03 | Replicas of one tool pod share L2 keys; two tool pods do not | P0 |
| TP-04 | The pod starts and stops the invalidation listener via `coll-task-01`'s API | P0 |
| TP-05 | The pod constructs no `SQLiteBackend` directly | P0 |
| TP-06 | The pod opens the collections bucket eagerly at startup via `ensure_kv_bucket`, before its registry is configured (`coll-task-04a` KVC-05) | P0 |
| TP-07 | Proven live: a second principal is **refused** on this pod's keys | P0 |

---

## The stack builder

A tool pod should call one function and get its tiers, constructing no
`SQLiteBackend` -- the cache-primitive allowlist is per-repo and cannot see a
partner-operated fourth repo, so the builder makes that question moot rather than
exempted. `tests/enforcement/test_no_bespoke_reuse.py` already flags the
"stores an `SQLiteBackend` + cache verbs without subclassing `BaseCollection`"
shape, which is most of TP-05.

**Home: `ToolServerBootstrap` in `threetears.agent.tools.bootstrap`**, not
`ToolServer`. Bootstrap's own docstring says it owns the canonical tool-pod
lifecycle -- it was written to end three drifted copies of start/stop scaffolding
and already wires the health surface, which is where TP-04's listener start/stop
belongs. `ToolServer` is the MCP request handler and owns no cache concern.

**Dependency: `3tears-agent-acl[bus]` was NOT added, and the shard was wrong to
ask for it.** The reasoning it gives -- "`AclInvalidationSubscriber` degrades to
plain `object` without the extra" -- is about the ACL invalidation bus
(`{ns}.acl.invalidate.*`), which is an RBAC-cache concern belonging to the
registry, the SDK and the hub. A tool pod's collection stack touches none of it:
`CollectionRegistry.start_invalidation_listener` lives in `threetears.core` and
subscribes the cross-platform `threetears.cache.invalidate` subject, so nothing
in this landing imports `threetears.agent.acl` at all. Declaring it anyway would
have been a stale dependency, and `tests/enforcement/test_dependency_alignment.py`
(declared deps must match actual imports) would have failed the build for it.

**This would be the third stack builder** (`build_three_tier_stack`,
`build_registry_rbac_stack`). `coll-task-07a` consolidates only the ACL-subscriber
half. State the intended end state in the PR rather than leaving three.

---

## Files to Modify

- `packages/nats/src/threetears/nats/subject_permissions.py` -- `_tool_pod`'s buckets, scope and invalidation grant.
- `packages/agent/tools/src/threetears/agent/tools/bootstrap.py` -- the collection-stack builder and the listener start/stop.
- ~~`packages/agent/tools/pyproject.toml` -- the `3tears-agent-acl[bus]` dependency.~~ **Not needed; see above.**
- `packages/nats/tests/integration/test_user_jwt_scoped_grant_live.py` -- the refusal probe for TP-06.


---

## The proving case

`dipp-tool-server` is **not in any checkout** -- removed in PR #340 and named only
in prose. Prove TP-06 against a tool pod in this workspace. The pentest pod's
lifecycle is pointed at from `14-eng-ai-bot/docs/runbook-platform-cold-start.md`
Step 8 and `docs/DEPLOYMENT.md` §11b; both defer to the standalone runbook in the
`14-eng-ai-bot-agent-pentest` repo, which is the actual home.

---

## Anti-patterns

- DO NOT scope by a namespace id. One row per tool, pod-influenced, deliberately colliding, absent at connect.
- DO NOT add plumbing to carry a scope. It is already `claims.sub`.
- DO NOT grant `$KV.` on subscribe. It confers no read and leaks every value.
- DO NOT add the collections grant without resolving TP-02.
- DO NOT put the builder on `ToolServer`. Bootstrap owns the lifecycle.
- DO NOT roll a bespoke SQLite cache, and do not add a per-repo enforcement exemption for one.
- DO NOT assume a missing grant hangs. Through `kv_bucket` it raises.

---

## Success criteria

- [x] `_tool_pod` grants the scoped collections bucket and the recorded invalidation grant -- 
      `JsResource.kv(f"{ns}-collections", scope=kv_key_scope_for(TOOL_POD, pod_id=p), writable=True)`,
      plus `CROSS_PLATFORM_CACHE_INVALIDATE` on BOTH directions
- [x] Two pods with different `tool_pods.id` produce disjoint keys; two replicas of one produce
      identical ones -- `TestTheToolPodHoldsTheSharedBucket` (grant side) and
      `TestTheStackIsScopedToTheToolPodId` (key side)
- [x] The pod constructs no `SQLiteBackend`; the sanctioned factory is
      `threetears.agent.tools.l1_cache.create_tool_pod_l1_backend`, declared in
      `test_cache_primitive_usage.py`'s `allowed_sqlite_construction_sites`. **Read the caveat
      below** -- that walker's root discovery does not currently reach `packages/agent/*`
- [x] ~~The acl dependency is declared~~ -- WITHDRAWN on evidence; see "The stack builder"
- [x] TP-02's decision is written into the ledger's Part 4 -- it already was, verbatim
- [x] `./scripts/check-all.sh` (16030 passed, 3 skipped, 412 deselected; 139 sidecar) and
      `./scripts/test-integration.sh` (393 passed, 19 skipped) green

---

## Verification

1. Bring up an in-workspace tool pod with a real two-tier collection.
2. Write through replica A; read through replica B; B serves the L2 copy.
3. Write through A again; B's L1 **and** L2 are evicted (see `coll-task-03` -- L2
   eviction is part of the scoping landing).
4. From a **different principal's** credential -- callout-minted **and** static -- 
   attempt to read this pod's keys. Refused, both direct and body-carried.
5. `nats kv ls` shows the pod's keys under its own scope and nowhere else.

Step 4 is TP-07, and it is what proves the sequence did its job. Without it this shard has shown
only that a tool pod can cache, not that it can cache safely.

**Steps 1-3 and 5 are NOT run.** They need a live cluster, and the hub cannot boot until the whole
`-03`-onward block lands together (`coll-task-06a` and `-04a` record the same gap for the same
reason). Step 4 IS run -- see below.

---

## What landed differently from the shard

- **TP-07 is met by `test_a_tool_pods_keys_are_refused_to_every_other_principal`**, in
  `packages/nats/tests/integration/test_user_jwt_scoped_grant_live.py`, against a real
  `nats-server` testcontainer. THREE credentials on one shared bucket, because the two refusals
  fail differently: a MINTED peer tool pod (a second `tool_pods.id`) is stopped by an allow-list
  that never names A's scope, while the STATIC `tool_server` user is stopped by a generated DENY
  layered under a coarse `$KV.>` allow that would otherwise cover everything. The `allow_direct`
  premise is asserted before either -- with it false, every refusal would be vacuous -- and both
  admit-halves are asserted too: A reads and writes its own scope, and the static user reaches a
  bucket it is not denied, so "everything was refused" cannot be mistaken for a broken credential.
  The static user's deny set is GENERATED from `js_api_grants_for_stream`, never hand-typed.

- **A non-uuid tool-pod id can no longer be granted at all**, which is a behaviour change the shard
  did not name. `coll-task-05a` landed GRANT-10 at the resolver and recorded that "a tool pod is
  unaffected until `coll-task-07c` gives it the bucket"; giving it the bucket makes `_tool_pod`
  derive a scope, and `kv_key_scope_for` refuses anything non-uuid. Two live integration modules
  were carrying slug pod ids (`pod-alpha`, `pod-A`) and now carry uuids.

- **The eager bind was PROMOTED, not copied.** `coll-task-06a` put a bounded-retry
  `ensure_kv_bucket` loop in `RegistryServer.open_collections_bucket`; the hub carries a third copy
  in `hub/common/collections_bucket.py`. Rather than write a fourth, the loop moved to
  `threetears.core.collections.bucket.bind_collections_bucket` and the registry server now
  delegates to it. It lives in `core` rather than `nats` because the bucket NAME is
  `BaseCollection.L2_BUCKET_SUFFIX`, and `nats` may not depend on `core`. **The hub's copy is still
  its own** -- a hub edit this shard may not make.

- **`ToolServer` gained one narrow lifecycle seam, `add_connected_callback`.** The bootstrap owns
  the stack (as the shard requires) but cannot build it without the connection, and `ToolServer`
  opens that connection inside `serve()` and deliberately does not expose the client. Injecting a
  pre-connected client instead would have flipped `_owns_nats_connection` and silently disabled the
  pod's proactive NATS-JWT re-auth loop. The hook runs after connect and BEFORE the call-subject
  subscribe and the registration publish, so a pod is never discoverable with its collections
  unwired; that ordering is pinned by
  `test_connected_callbacks_run_before_the_pod_is_reachable`.

- **A new enforcement gate, `tests/enforcement/test_eager_collections_bucket_open.py`**
  (`EAGER_BUCKET_OPEN_ENFORCEMENT_MODE`, default `strict`;
  `_eager_bucket_open_exemptions.txt`). Nothing statically required KVC-05 of the NEXT process to
  wire an L2 registry, and the omission is silent in both directions. The rule: a module that binds
  the registry-default L2 client must also name an eager opener.

- **A real blind spot in the shared enforcement helper, found here and FIXED later on this
  same branch.** Commit `b81c0beb` ("Stop the enforcement walkers being blind to a third of
  this repo") took the landing this bullet describes, including the ten genuine violations
  the widening surfaces. The description below is why the shard did not take it, and it is
  history: do not read it as live work, and do not re-derive the blind spot from it.

  `threetears.enforcement.common.find_local_src_roots` walked `packages/*/src` only, so on this repo
  it silently returned nothing for the ten nested `packages/agent/*` packages -- including
  `agent/tools`, where this shard's code lives. Every walker built on it (`test_kv_grant_capability`,
  `test_l2_scope_wiring`, `test_cache_primitive_usage`, `test_no_bespoke_reuse`,
  `test_underscore_access`, …) had been reporting a clean tree over a third of the repository.
  Widening it was implemented and reverted: it immediately surfaced **10 genuine violations** in
  five packages (8 `cache.missing_collection` for `identity_versions`, `intentions`,
  `memory_consolidations`, `agent_skills`, `agent_skill_invocations`, `agent_wake_schedules`,
  `wake_fires`, `webhook_subscriptions`; 1 `cache.pool_access` on `memory_chunks` in
  `agent/memory/tools.py`; 4 `underscore_access.E` in `agent/wake`) plus two stale fixtures in
  `packages/enforcement/tests/common/test_repo_layout.py`. That is a landing with its own review,
  not a side effect of a grant shard. **This shard's own gate composes the nested roots itself**
  (`_scan_roots`, with the reasoning in its docstring) so it is not blind to the module it exists
  for, and `test_the_rule_is_not_vacuous` names `packages/agent/tools/.../bootstrap.py` explicitly
  so the gate cannot silently stop covering it. The tool-pod L1 factory is declared in
  `allowed_sqlite_construction_sites` anyway, so closing the blind spot will not turn a correct
  factory into a violation.

- **TP-06 vs TP-07 numbering.** The shard's requirements table has seven rows: TP-06 is the eager
  bucket open and TP-07 is the refusal probe. Its "Files to Modify" section calls the probe
  "the refusal probe for TP-06". Both requirements are met; the table's numbering is the one used
  here.
