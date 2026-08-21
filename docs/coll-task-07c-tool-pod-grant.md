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

`_tool_pod` grants exactly two KV buckets — `{ns}-proxy_assertion_nonces` and
`{ns}-leases`. **No `{ns}-collections`.**

`CROSS_PLATFORM_CACHE_INVALIDATE` appears in the pub and sub tuples of the agent
pod, registry, hub, gateway and channel adapter, **and in neither of the tool
pod's**. So a tool pod can neither announce a write nor learn of one.

The tool pod is one of two principals missing the collections grant; the registry
is the other, and `coll-task-05a` fixes it.

### What a missing grant does — three cases

- **raw JS control-plane call on an ungranted subject**: does not raise. Blocks to
  the deadline and reads as an unreachable broker.
- **bucket open through `KVLease`**: **raises `KvError`** after a JetStream
  timeout, which nothing catches — a hard failure on first claim.
- **bucket open through a `BaseCollection`** — which is what this shard is about:
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

A namespace-derived scope was designed and rejected on evidence — worth recording
because the reasoning generalizes:

- `tool_namespace_id` mints **one row per tool** — see the ledger's Part 2 for
  the counts and their anchors. The shape is what matters, not the number.
- It is a pure function of manifest values **the pod sends**, bounded only by an
  `allowed_namespaces` prefix test — so a pod could mint unbounded new scopes into
  a memory-backed bucket with no `max_bytes`.
- Its docstring says platform pods deliberately **collide** on one row, which is
  the opposite of the isolation property a scope must have.
- Rows are written by `tools.register`, **after** the JWT is minted, so no such
  value exists at connect.

An earlier draft of this shard carried the anti-pattern *"DO NOT scope by
`pod_id`. Every replica gets its own cache."* That premise was never verified and
is **false for tool pods**: `tool_pods.id` is per-deployment, not per-process.

### Multi-namespace pods: answered

Source answers it — every platform tool pod serves many namespaces. An earlier
"if none does, refuse" branch would have refused the platform's own tool servers.
Scoping by `tool_pods.id` makes the question moot; the collection does not need
to know which namespace it belongs to.

---

## The invalidation grant is a decision, not a detail

Granting a partner-operated pod pub and sub on `threetears.cache.invalidate` — a
single non-namespaced global subject carrying `{table, ids, origin}` with the raw
pk of every write — gives it:

- a **cross-customer metadata firehose**: every table name and entity id written
  by every agent, hub replica, gateway and adapter;
- a **fleet-wide eviction primitive**: forged messages evict L1 everywhere, and
  with no L1 max age set anywhere that is an unbounded stampede onto YugabyteDB.
  `_on_invalidation` filters only on self-asserted `origin` — trivially spoofed by
  asserting a different one — and drops the **RBAC scan cache before** the
  collection, L1 and has-table guards, so a forged message naming any table drops
  a security cache on pods that hold no such collection.

**The decision, so an implementer does not have to make it:** grant the global
subject, record the exposure as accepted for this landing, and defer both
`origin` authentication and any move to a scoped subject.

The reasoning: the static NATS users already hold this subject unprefixed, so a
tool pod is not the widest holder and scoping it alone would buy little
(closing that side is `coll-task-05b`'s). Both
deferred items are publisher-side changes across three repos — a wire-protocol
change, not a grant change — and they need their own shard with their own
review. Carrying them as a P0 checkbox here would make the implementer the
ratifier of a security posture.

**`coll-task-08-invalidation-origin-auth`** owes two things: `origin` minted from
the authenticated principal, and `_on_invalidation` rejecting a table the
receiver does not hold **before** touching the scan cache. It is not written yet;
the name exists so the deferral is findable rather than floating. Note the guard reorder overlaps
`coll-task-03`'s L2S-09, which restructures the same function — sequence them or
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
`SQLiteBackend` — the cache-primitive allowlist is per-repo and cannot see a
partner-operated fourth repo, so the builder makes that question moot rather than
exempted. `tests/enforcement/test_no_bespoke_reuse.py` already flags the
"stores an `SQLiteBackend` + cache verbs without subclassing `BaseCollection`"
shape, which is most of TP-05.

**Home: `ToolServerBootstrap` in `threetears.agent.tools.bootstrap`**, not
`ToolServer`. Bootstrap's own docstring says it owns the canonical tool-pod
lifecycle — it was written to end three drifted copies of start/stop scaffolding
and already wires the health surface, which is where TP-04's listener start/stop
belongs. `ToolServer` is the MCP request handler and owns no cache concern.

**Dependency:** add `3tears-agent-acl[bus]` to `packages/agent/tools`. No cycle —
acl does not depend on tools. The extra pulls `3tears-nats[client]`, which tools
already has, so the only genuinely new package is acl itself. The extra is
required because `AclInvalidationSubscriber` degrades to plain `object` without
it.

**This would be the third stack builder** (`build_three_tier_stack`,
`build_registry_rbac_stack`). `coll-task-07a` consolidates only the ACL-subscriber
half. State the intended end state in the PR rather than leaving three.

---

## Files to Modify

- `packages/nats/src/threetears/nats/subject_permissions.py` — `_tool_pod`'s buckets, scope and invalidation grant.
- `packages/agent/tools/src/threetears/agent/tools/bootstrap.py` — the collection-stack builder and the listener start/stop.
- `packages/agent/tools/pyproject.toml` — the `3tears-agent-acl[bus]` dependency.
- `packages/nats/tests/integration/test_user_jwt_scoped_grant_live.py` — the refusal probe for TP-06.


---

## The proving case

`dipp-tool-server` is **not in any checkout** — removed in PR #340 and named only
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

- [ ] `_tool_pod` grants the scoped collections bucket and the recorded invalidation grant
- [ ] Two pods with different `tool_pods.id` produce disjoint keys; two replicas of one produce identical ones
- [ ] The pod constructs no `SQLiteBackend`; `test_no_bespoke_reuse.py` passes with no new exemption
- [ ] The acl dependency is declared
- [ ] TP-02's decision is written into the ledger's Part 4
- [ ] `./scripts/check-all.sh` and `./scripts/test-integration.sh` green

---

## Verification

1. Bring up an in-workspace tool pod with a real two-tier collection.
2. Write through replica A; read through replica B; B serves the L2 copy.
3. Write through A again; B's L1 **and** L2 are evicted (see `coll-task-03` — L2
   eviction is part of the scoping landing).
4. From a **different principal's** credential — callout-minted **and** static —
   attempt to read this pod's keys. Refused, both direct and body-carried.
5. `nats kv ls` shows the pod's keys under its own scope and nowhere else.

Step 4 is TP-07, and it is what proves the sequence did its job. Without it this shard has shown
only that a tool pod can cache, not that it can cache safely.
