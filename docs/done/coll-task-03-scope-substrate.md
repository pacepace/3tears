# coll-task-03: L2 Key Scoping Substrate (3tears)

## Objective

Give the shared `{ns}-collections` NATS KV bucket intra-bucket key isolation, so
`coll-task-05a` can narrow the grant from `$KV.{bucket}.>` to a per-principal
subject.

This lands the data half: the key shape, the scope helper, and validation at
wiring time. Facts cited below live in the evidence ledger
(`14-eng-ai-bot/docs/collection-support-evidence.md`).

---

## The problem

`BaseCollection.l2_key` returns `{table_name}.{body}`. Every collection in every
principal writes into one bucket with no principal segment, so `mint_user_jwt`
can only grant `$KV.{bucket}.>`.

`user_jwt.py`'s intra-bucket-isolation comment documents this as a deliberate
residual and names the blocker:

> Tightening to ``$KV.{bucket}.{prefix}.>`` is impossible without a key-prefix
> the data layer does not write; it would break every read.

**This shard supplies that prefix — for the collections bucket only.** The
comment names three buckets; the other two are recorded as out of scope in
`coll-sequence.md`.

---

## Every key is scoped. There is no shared tier.

One tier, `{scope}.{table}.{body}`, always.

A two-tier design with an opt-in `SHARED` scope was worked through and dropped.
The ledger's Part 4 carries the reasoning; the short form is that a shared tier
cannot be made read-only. `$KV.` grants are pub-and-sub with no split,
`_pull_through` writes L2 on every miss so no principal is ever a pure reader,
and the tables nominated for sharing carry `customer_id`.

Per-principal copies of a few thousand reference rows in a memory-backed bucket
are cheaper than that problem. Do not reintroduce a shared tier without
answering all three.

---

## Design context

**L2 is read-through** — a miss falls to `fetch_from_store` — so for a collection
with an L3 tier a key that no longer resolves costs a fetch, not an answer.

That is the only safety property scoping gets for free. An earlier draft claimed
a second one — *"invalidation does not key on the KV key, so re-scoping cannot
break cross-pod eviction"* — which is true of the **message** and false as a
safety argument. See below.

**Neither holds for a collection with `L3 = None`**, where L2 *is* the source of
truth and a key that no longer resolves is data that no longer exists. The
ledger's Part 2b enumerates them. The dangerous one is
`IdentityGenerationCollection`: the identity fence **fails open** — no known
generation grants — so a lost key admits a superseded connection rather than
refusing it.

That collection is **PRIVATE**, decided here rather than deferred: writer and
reader are both the hub across replicas — the class docstring says a handshake
completed on one replica must fence a connection whose callout resolves on a
sibling, and the `hub/app.py` call site reinforces it with "ONE instance shared
by both".

**The cutover is `coll-task-06b`'s, and it is not a free cache miss.** Re-keying
disarms the fence for every in-flight reauth. Do **not** improvise a remedy here:
the two obvious ones — draining connections, or flipping the fence to
fail-closed-on-unknown — are both unsafe, for reasons `-06b` sets out. Follow
that shard.

### Scoping breaks L2 coherence unless invalidation also evicts L2

This is the defect scoping introduces, and it must land in the same commit.

`_on_invalidation` drops the RBAC scan cache and calls `l1.delete_by_id`. **It
never deletes L2.** Today that is fine because every principal shares one key:
the hub writes it, and a peer's `_pull_through` — which reads L2 before L3 —
reads the writer's fresh value.

After scoping, each principal has its own key. The hub revokes a role, updates
**the hub's** key, publishes invalidation. The peer drops L1, pulls through, hits
**its own stale key**, and re-caches the revoked grant. `max_age` is unlimited
and `set_l1_max_age` has zero production callers, so nothing ever heals it.

A revoked grant enforced forever is worse than the exposure this sequence exists
to close. So `_on_invalidation` must delete the receiver's **own** scoped L2
entry for `(table, ids)`.

The mechanism works: a KV delete is a `js.publish` to
`$KV.{b}.{scope}.{table}.{body}` with a delete-marker header, so it is covered by
the per-principal publish grant, and `deny_delete: true` gates
`$JS.API.STREAM.MSG.DELETE` rather than the marker.

**FOUND IN IMPLEMENTATION, and not in the requirement above: the eviction must not run
on a collection with `L3 = None`.** As written, L2S-09 destroys data. Two ways:

- *Same principal.* Replicas share a scope and therefore share a key. Replica A writes
  and publishes; replica B evicts "its own" key, which IS the key A just refreshed. With
  L3 present that costs a pull-through and the row comes back. With `L3 = None` the row
  is gone — proven by `TestHeartbeatCollectionL2Coherence`, whose peer read returned
  `None` the moment the eviction landed.
- *Across principals.* Each principal's key under `L3 = None` is its own source of truth,
  not a stale view of somebody else's, so there is nothing there to invalidate either.

The staleness the eviction exists to prevent also cannot arise without an L3: with no
`fetch_from_store` behind it, nothing re-caches anything. So `delete_l2_entry` returns
early when `self.l3_pool is None`. That is the same predicate, for the same stated reason,
that `BaseCollection.l1_max_age_seconds` already applies to L1 expiry — "a tier that is
the source of truth is not a cache" — and the ledger's Part 2b enumerates exactly the
collections it protects, `IdentityGenerationCollection` (fails open on a missing
generation) foremost.

Three further implementation constraints, each of which would silently defeat it:

1. **Place it immediately after the `collection is None` return**, before the L1
   backend is fetched. Putting it "before the L1 delete" leaves it behind the
   `l1 is None` and `not l1.has_table(...)` guards — and that `has_table` guard
   exists precisely for tables whose L1 schema was never initialized, so those
   collections would skip the L2 delete and keep the revoked grant forever. L2
   presence is independent of L1 presence.
2. **Promote a public accessor.** The eviction needs `_delete_from_l2` from
   `registry.py` — cross-class private access, which `SLF` in `lint.select`
   forbids repo-wide and which the underscore contract says to fix by promoting,
   not exempting. Add `BaseCollection.delete_l2_entry(entity_id)`, public, with a
   Sphinx docstring. (It is a METHOD, so "and in `__all__`" does not apply —
   `base.py`'s `__all__` carries module-level names and already exports
   `BaseCollection`. What the underscore contract asks for here is the absent
   leading underscore, which is what makes the cross-class call legal.)
   `_delete_from_l2` STAYS, private and unconditional: the write path's delete
   must land even against a racing write, so it cannot pay for the presence probe
   below. The two have different contracts, not parallel paths.
3. **Gate on L2 presence.** A KV delete writes a marker **unconditionally**, so
   an ungated eviction writes markers for entities the receiver never cached,
   into a memory bucket with `history=1`, unlimited `max_age` and no `max_bytes`.
   Check presence first. (Bounding `max_age`/`subject_delete_marker_ttl` on the
   stream was the alternative; `coll-task-04a` compares only `direct` and asserts
   unlimited `max_age` throughout, so it does not carry that requirement and this
   shard must not assume it will.)

### The scope is the sharing boundary

Replicas of one principal must land on one key or L2 stops being a cross-pod
cache.

| principal | scope input | why |
|---|---|---|
| agent pod | `agent_id` | authenticated from the identity JWT's `sub`; replicas share |
| tool pod | `tool_pods.id` | its authenticated `claims.sub`; configured once per deployment, shared by replicas |
| infra services | the `Principal` enum value | one identity per service; there is no per-connection id to use |

**Never `conn_id`** — it is per-connection, so every reconnect orphans the cache.

**And never `pod_id` for an agent pod.** For `AGENT_POD` the callout sets
`_CLAIM_POD_ID` to `_safe_segment(connect_name) or agent_id` — the connect name
is **attacker-influenced**. `kv_key_scope_for` must therefore refuse `pod_id`
for `AGENT_POD` and accept only `agent_id`. A helper that takes a spoofable
input for one principal and an authenticated one for another is the shape this
whole sequence exists to avoid.

For a tool pod the same claim is safe, because `_resolve_tool_pod` ignores the
connect name and pins `claims.sub` from the verified key id.

### The cutover

The bucket is memory storage with no durable data. A NATS restart already
discards every entry and the platform survives. No migration, no dual-read, no
shim.

**Do not plan around TTL expiry** — `max_age` is unlimited, so old-shape keys do
not age out while the process lives.

---

## Package placement

The scope constant and helper go in **`packages/nats`**, not core.

Core depends on `3tears-nats`; nats depends only on `3tears-observe` and
pydantic. A grant builder in `user_jwt.py` importing from `threetears.core`
would be a dependency cycle and would drag core's stack into a package that
deliberately installs without it.

`subject_permissions.py` already hosts this exact shape:
`CROSS_PLATFORM_CACHE_INVALIDATE` is a wire constant both sides read, and
`inbox_prefix_for` is a pure identity-to-subject helper called by each
per-principal resolver on the mint side and by the connecting process on the
client side. Its SDK call-site comment documents the failure mode to avoid:
keying the client on a different value than the mint side produces a silent
timeout on a subject the JWT does not grant.

So `kv_key_scope_for(principal, *, agent_id=None, pod_id=None)` sits
beside `inbox_prefix_for`, and `base.py` imports it.

`subject_permissions` is in the **eager** import block of
`threetears/nats/__init__.py`, so new constants there stay importable without
pulling nats-py — the L1-only install property survives.

**An infra principal's identifying id is the enum value itself** — `HUB`,
`REGISTRY`, `GATEWAY`, `CHANNEL_ADAPTER`, and the two `coll-task-05a` adds. So
`kv_key_scope_for(Principal.REGISTRY)` with no further argument is legal and
returns a stable literal. L2S-08's "may not collide with a bare `Principal`
value" therefore constrains **pod-derived** scopes only; it must not be
implemented as a blanket ban, or every infra principal is refused.

`kv_key_scope_for` **raises** for a *pod* principal missing its identifying id. A
fallback to the bare enum value would land every tool pod that loses its
namespace on one shared scope.

---

## The sanitizer: promote one, and stop using it for scopes

The dots-to-dash rule exists three times: public `sanitize_segment` in
`core/namespaces.py`, and private `_sanitize` in `subjects.py` and `_seg` in
`subject_permissions.py`. The grant side cannot reach core's copy, and importing
either private one is a Shape-A underscore violation.

**Promote `subjects._sanitize` in place, as public
`sanitize_subject_segment(value: str | UUID) -> str`, and have
`core.namespaces.sanitize_segment` delegate to it.**

Two constraints fix the home and the name:

- **It must live in `subjects.py`, not `subject_permissions.py`.**
  `subject_permissions` already imports from `subjects`; putting the promoted
  function in the former and rewiring the latter to it is an import cycle at
  module load.
- **It must not be called `sanitize_segment`.** A fourth copy of this rule
  already exists as public `threetears.media.contracts.keys.sanitize_segment`,
  with *different* semantics — it slugifies. Three public symbols of one name
  with three behaviours across three packages is worse than the duplication.

The signature must accept UUIDs: `_sanitize` has ~40 references across ~30 lines
in `subjects.py` — 34 call sites across 24 lines — several passing UUIDs, which would fail mypy under `str`-only.
Widening `core.namespaces.sanitize_segment` the same way breaks nothing — its
only non-test caller is `build_namespace_name`.

**But do not derive a scope from a sanitized name.** `sanitize_segment` is
`value.replace(".", "-")` and is therefore non-injective; `core/namespaces.py`
already records in source that two distinct mcp names can collapse onto one
name. Two tool pods sharing a scope is precisely the outcome this work prevents.
Scopes come from UUID hex. The sanitizer is for display-ish segments, not
security boundaries.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| L2S-01 | `l2_key` emits `{scope}.{table}.{body}`; the body grammar check and SHA-256 fallback are unchanged | P0 |
| L2S-02 | `configure()` raises when registry state holds an L2 client and no `kv_key_scope` | P0 |
| L2S-03 | Two principals never collide on a key; two replicas of one principal always do | P0 |
| L2S-04 | The scope segment is validated against `^[-_=a-zA-Z0-9]+$` — stricter than the key body grammar | P0 |
| L2S-05 | A backstop raise in `l2_key` covers the `nats_client=`-direct path, and is **not** a `KvError` subclass | P0 |
| L2S-06 | `kv_key_scope_for` raises for a **pod** principal with no identifying id, and refuses `pod_id` for `AGENT_POD` | P0 |
| L2S-07 | Only the collections bucket is scoped; `l2_key` for other buckets is unchanged | P0 |
| L2S-08 | A **pod-derived** scope may not collide with a bare `Principal` enum value; infra scopes *are* those values | P1 |
| L2S-09 | Invalidation deletes the receiver's own scoped L2 entry, not only L1 — and never on a collection with `L3 = None`, where L2 is the source of truth and eviction is deletion | P0 |

---

## Fail at wiring time

`l2_key` is called only from `_get_from_l2`, `_save_to_l2`, `_delete_from_l2`
and `l2_cas_mutate` — all first-read/first-write paths. A raise there does not
make a process "fail at startup"; it makes it die on the first cache access under
load, which for the agent router's sticky routing is a production outage.

So the check is at `configure()` (L2S-02), with `l2_key` as the backstop for the
construction path that never calls it. After this change `l2_key` reads the scope off `self._registry` — a required
positional on `BaseCollection.__init__` — which is also why the backstop only
fires when the registry itself is unscoped. (Today it touches `_registry` not at
all.) Do not add a constructor parameter.

**L2S-02 is evaluated over registry state after the merge, not over this call's
arguments.** `configure()` merges — `if l2_client is not None: self._l2_client = ...`.
`coll-task-02` establishes that two-pass wiring (scope first, client later, or
the reverse) is the normal shape at several sites. A naive per-call check breaks
every one of them.

**The backstop must not raise `KvError`.** Three of the four `l2_key` call sites
sit inside `except KvError` handlers that degrade to a warning, so a `KvError`
would be swallowed and the fleet would run with L2 silently off — the degradation
this decision exists to prevent. `coll-task-04a` makes the same point for its own
exception type.

The fourth, `l2_cas_mutate`, deliberately does **not** degrade — its docstring
says L2 is the source of truth there and the error must propagate. So a `KvError`
backstop would be swallowed at three sites and propagate at the one where L2 is
authoritative: inconsistent in the worst direction. A distinct type is consistent
at all four.

The degrade alternative — warn once, serve from L1+L3 — was considered and
rejected: it silently loses cross-pod coherence and is found months later by a
stale-read bug that reproduces on nothing.

---

## The scope grammar is stricter than the key grammar

`_KV_KEY_GRAMMAR` at `base.py:55` is `^[-/_=.a-zA-Z0-9]+$` — **`.` is inside the
character class.** Reusing it for a scope validates nothing about dots and would
ship a check that accepts a value producing two subject tokens, silently
defeating the grant. It also permits `/`, and leading, trailing and doubled dots.

---

## Files to Modify

- `packages/nats/src/threetears/nats/subjects.py` — promote `_sanitize` to public `sanitize_subject_segment`; rewire its 34 in-file call sites.
- `packages/nats/src/threetears/nats/subject_permissions.py` — `kv_key_scope_for`, the scope grammar; `_seg` delegates to the promoted sanitizer.
- `packages/nats/src/threetears/nats/__init__.py` — add the new names to the eager re-export block **and** `__all__`. While here: its comment claims `test_lazy_surface` checks the TYPE_CHECKING block; it does not (see below).
- `packages/nats/tests/unit/test_lazy_surface.py` — it asserts lazy-map ↔ submodule exports and lazy-map ⊆ `__all__`. It never parses the TYPE_CHECKING block, despite the test named for it.
- `packages/core/src/threetears/core/collections/base.py` — the scope segment in `l2_key`, the backstop raise, and the promoted public `delete_l2_entry`.
- `packages/core/src/threetears/core/collections/registry.py` — `kv_key_scope` on `configure()`, the state-based refusal, and L2S-09's L2 eviction in `_on_invalidation`.
- `packages/core/src/threetears/core/namespaces.py` — `sanitize_segment` delegates.
- `packages/core/tests/test_base_collection.py` — `TestL2KeyGrammarSafe` updated for the prefix; it also holds most `set_l1_max_age` calls.

### Create
- `packages/core/tests/test_l2_key_scoping.py` — the spec test, per the contract below.

---

## The spec test

A draft of this file existed and was **deleted**: it was written against the
two-tier PRIVATE/SHARED design that Part 4 dropped, so it declared an `L2Scope`
enum this design does not have and asserted an unscoped registry *does not*
raise. Left in place it would have specified the wrong contract while its own
success criterion read green.

Write it fresh. The contract:

- a collection with no declaration keys `{scope}.{table}.{body}`;
- two scopes never collide; two registries on one scope always agree;
- `configure(l2_client=X)` with no scope raises, naming `kv_key_scope`;
- scope-then-client and client-then-scope both succeed (the two-pass wiring
  above), because the check is over merged registry state;
- a scope containing `.` or `/` is refused at `configure()`;
- a **pod-derived** scope colliding with a bare `Principal` value is refused —
  and, in the same test, `kv_key_scope_for(Principal.REGISTRY)` is **accepted**,
  because an infra scope *is* that value. `configure()` cannot tell the two apart
  on its own, so the check belongs in `kv_key_scope_for`, not in `configure()`;
- the backstop raise from `l2_key` is not a `KvError` subclass;
- the scope segment survives body hashing — assert only that delta; the SHA-256
  fallback invariant belongs to `test_base_collection.py`;
- invalidation evicts the receiver's own scoped L2 entry (L2S-09).

There is no `L2Scope` enum in a single-tier design. If one appears in
implementation, the tier decision has been reopened by accident.

---

## Implementation notes

1. **`l2_key` composes.** Compute `{table}.{body}` exactly as today, then prefix.
2. **`kv_key_scope` joins `configure()`'s keyword arguments**, not the
   constructor. There are `CollectionRegistry()` construction sites across three
   repos (none in admin), though most are L3-only and need no scope.
3. **Do not hash the scope segment.** It must stay readable in the grant or an
   operator cannot tell which principal a subject belongs to.
4. **Update `l2_key`'s docstring.** It documents the old shape in three places.

---

## Anti-patterns

- DO NOT put the scope constant or helper in core. The dependency direction makes the grant-side import impossible, and the implementer will resolve it by hand-typing the value.
- DO NOT reuse `_KV_KEY_GRAMMAR` for the scope. It permits the dot.
- DO NOT derive a scope from `sanitize_segment`. It is non-injective and source already records the collision.
- DO NOT let the backstop raise a `KvError`. It would be swallowed.
- DO NOT check L2S-02 against one call's arguments. Two-pass wiring is normal.
- DO NOT scope buckets other than collections. Nothing else writes a prefix, and a missing `$KV.` match does not raise — it blocks to the deadline and reads as an unreachable broker.
- DO NOT reintroduce a shared tier. Three problems, all in the ledger.
- DO NOT implement L2S-09 by calling `invalidate_cache` from `_on_invalidation`. It is public, it does look exactly right — and it **re-publishes**. Since the `origin` filter only skips *self*, every receiver would rebroadcast under its own origin: an unbounded eviction storm. Use the promoted `delete_l2_entry`.
- DO NOT put the blanket `Principal`-collision ban in `configure()`. It cannot distinguish a pod-derived scope from an infra one, and would refuse every infra principal at startup.

---

## Success criteria

- [x] The spec test is written to the contract above and passes; no `L2Scope` enum exists
- [x] Invalidation evicts the receiver's own scoped L2 entry, proven by a two-principal test: hub revokes, peer refuses
- [x] `TestL2KeyGrammarSafe` updated and passing
- [x] A scope containing `.` or `/` is refused at `configure()`
- [x] `configure(l2_client=X)` with no scope raises, naming `kv_key_scope`; scope-then-client and client-then-scope both succeed
- [x] The backstop raise is not caught by `_get_from_l2` / `_save_to_l2` / `_delete_from_l2`
- [x] `kv_key_scope_for` importable from `threetears.nats`; `test_lazy_surface` green
- [x] `./scripts/check-all.sh` green; `./scripts/test-integration.sh` green

---

## Verification

```bash
cd 3tears
./scripts/test.sh core -v
./scripts/check-all.sh
./scripts/test-integration.sh
```

Use the scripts — `3tears/CLAUDE.md` forbids running pytest, ruff or mypy
directly. `check-all.sh` excludes integration tests, which is where cross-pod
behaviour lives.

---

## Enforcement test suggestions

Build on `threetears.enforcement.common` (`Violation`, `iter_python_files`,
`parse_python_file`, `parse_exemptions_with_rationale`, `apply_exemptions`,
`emit_report`, `resolve_mode`) and follow the established mode-env-var plus
`_*_exemptions.txt`-with-rationale shape. `tests/enforcement/test_no_bespoke_reuse.py`
is the worked example; re-rolling a walker here would break the rule this set
enforces.

- [x] **No scope is derived from a sanitized display name.** Suggested: pin `kv_key_scope_for` as the only producer of a scope value, so a future principal cannot be added with a name-derived one. The collision is already documented in source and would otherwise be reintroduced by the next author.
- [x] **The backstop exception type appears in no `except` clause alongside `KvError`.** A one-rule walker; it is the specific way this decision gets silently undone later.

Both landed in `tests/enforcement/test_l2_scope_discipline.py`, over
`threetears.enforcement.common`, mode `L2_SCOPE_ENFORCEMENT_MODE` (default `strict`),
exemptions in `_l2_scope_exemptions.txt` (deliberately empty). Each walker carries a
planted-violation self-test AND a planted-NON-violation test, so neither "matches
everything" nor "matches nothing" reads as a pass.
