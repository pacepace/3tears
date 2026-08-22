# coll-task-06a: Wire the Scope -- 3tears

## Objective

`coll-task-03` refuses an L2 client with no scope. Supply it at the 3tears
consumer, and add the enforcement rule the other two repos' shards rely on.

Split by repo because the three land at different points in the version dance
(`coll-sequence.md`, Landing mechanics) and because a single shard spanning
~40 sites in three repos is not one context window.

Site list and ratified decisions: the evidence ledger, Part 2b and Part 4.

---

## The sites -- two registries in one process

The registry server builds **two** `CollectionRegistry` instances, and an earlier
draft of this shard saw only the first:

- `registry/server.py` -- configures `l1_backend` + `l2_client`, starts the
  invalidation listener, and holds `HeartbeatCollection`.
- `rbac_stack.py` -- a second registry, `configure(l1_backend, l3_pool)` with no
  `l2_client`, then five ACL collections each passing `nats_client=` directly.
  Wired from `server.py`. Exact mirror of the hub's `gateway/acl.py` shape.

Both need a scope; both resolve to the registry principal.

`core/data/collection_factory.py` and `core/data/store.py` are library helpers
and own no scope; a process entry point does.

`HeartbeatCollection` is **`L3 = None`** and lives in the collections bucket, so
this process's L2 is a source of truth, not a cache. Coordinate with
`coll-task-05a`, which adds the collections grant the registry has never held.

**The registry currently runs its L2 collection against a bucket it is not
granted** -- see `coll-task-05a`. Wiring a scope here without that grant lands a
correctly-scoped key on a bucket the principal cannot reach.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| W3T-01 | The registry server sets `kv_key_scope` from `kv_key_scope_for(Principal.REGISTRY)` | P0 |
| W3T-02 | The scope input is the authenticated identity the process presents at connect, not local config | P0 |
| W3T-03 | The enforcement rule below exists and covers all **five** wiring shapes | P0 |
| W3T-04 | Both registries in the registry-server process are scoped -- `server.py` and `rbac_stack.py` | P0 |
| W3T-05 | The process opens the collections bucket eagerly at startup via `ensure_kv_bucket`, before `configure(l2_client=)`, so a `KvConfigMismatch` raises at wiring rather than in a request path (`coll-task-04a` KVC-05) | P0 |

---

## The drift the enforcement rule must catch

The key-side scope and the grant-side scope are the same string produced in two
places. If they diverge, every read misses silently, falls through to L3, and
nothing logs it: correct answers, a dead cache.

A pinned-pair test proves both sides call `kv_key_scope_for`. It **cannot** prove
they feed it the same input -- that residual closes only by sourcing the input
from the authenticated identity. State both halves in the test's docstring so the
next reader does not over-trust it.

The walker must cover **five** shapes: `configure(l2_client=)`,
`register(..., l2_client=)`, `bind_table(...)`, a collection constructor's
`nats_client=` keyword, and a **positional** NATS client -- `hub/geo/wiring.py`'s
`FeatureCache(registry, config, nats_client, None)` passes it positionally, and a
keyword-only walker misses it. A narrower walker reproduces the blind spot that
hid four processes from an early sweep of this work.

Build on `threetears.enforcement.common` (`Violation`, `iter_python_files`,
`parse_python_file`, `parse_exemptions_with_rationale`, `apply_exemptions`,
`emit_report`, `resolve_mode`) with the established mode-env-var and
`_*_exemptions.txt`-with-rationale shape;
`tests/enforcement/test_no_bespoke_reuse.py` is the worked example and its own
docstring notes it delegates rather than re-rolling a walker.

---

## Files to Modify

- `packages/registry/src/threetears/registry/server.py` -- scope, plus W3T-05's eager open
- `packages/registry/src/threetears/registry/rbac_stack.py` -- the second registry
- `packages/nats/tests/enforcement/` -- the pinned-pair test and the wiring walker (new files). They belong here, not in the hub: both halves the pair pins (`kv_key_scope_for` in `threetears.nats`, `l2_key` in `threetears.core`) live in this repo.

---

## Anti-patterns

- DO NOT set a scope inside a library factory to make a test pass.
- DO NOT use `conn_id`. Every reconnect orphans the cache.
- DO NOT let the grant side and the key side derive the scope from different inputs.

---

## Success criteria

- [x] Both registries in the registry-server process set a scope -- `build_heartbeat_collection_registry`
      (extracted from `RegistryServer._start_handlers`) and `build_registry_rbac_stack`, both
      `kv_key_scope_for(Principal.REGISTRY)`
- [x] The eager bucket open runs before `configure(l2_client=)` -- `RegistryServer.open_collections_bucket`,
      called from `serve()` immediately after connect and before `apply_rbac_factory`, which is what
      builds the second registry
- [x] The pinned-pair test exists, with its limitation documented -- 
      `packages/nats/tests/enforcement/test_kv_scope_pinned_pair.py`
- [x] The wiring walker covers **five** shapes, including the positional one -- 
      `packages/nats/tests/enforcement/test_l2_scope_wiring.py`, one planted-violation test per shape
- [x] `./scripts/check-all.sh` (15947 passed, 3 skipped, 411 deselected; 139 sidecar) and
      `./scripts/test-integration.sh` (392 passed, 19 skipped) green
- [ ] Live: **not run.** The registry cannot be exercised end-to-end until the hub boots, and the hub
      boots only once `coll-task-06b` supplies its own `kv_key_scope`

---

## Verification

```bash
cd 3tears
./scripts/check-all.sh
./scripts/test-integration.sh
```

Live: `nats kv ls` shows the registry's keys under its own scope segment and no
bare `{table}.{pk}` keys reappear.

---

## What landed differently from the shard

- **`l2_create_if_missing=False`, on BOTH registries and on the eager open.** `coll-task-04a`
  KVC-04 says not to set it `False` until KVC-10 lands, and KVC-10's proof is still partial
  (JetStream-level, not "kill NATS under a running hub"). It is set anyway, because for THIS
  principal the constraint is a distinction without a difference: `coll-task-05b` has already
  narrowed the static `registry` user to `$JS.API.STREAM.INFO.KV_{ns}-collections` with **no
  `CREATE`**, so the declare path can only ever issue a `STREAM.CREATE` the broker never answers
-- a permissions refusal arrives as a JetStream deadline, not an error -- and then fall through
  to the bind that was always going to succeed. `_reopen`'s self-heal, which is what KVC-04's
  caution protects, is therefore already unreachable here: after a NATS restart the declare path
  and the bind path fail identically, the declare path just takes a deadline longer.

  The bind path also buys something the declare path silently drops: `_bind_kv_stream` compares
  `allow_direct` and raises `KvConfigMismatch`, while the declare path reconciles only on
  err_code 10058 and otherwise falls through to the bind **with no comparison at all**. With
  `create_if_missing=True` this process could never raise the mismatch W3T-05 exists to surface.

- **The eager open is a BOUNDED RETRY, not a bare call.** The shard's one-line
  `await nc.ensure_kv_bucket(...)` is correct on `KvConfigMismatch` and wrong on `KvError`: the
  hub declares this bucket in its own lifespan and nothing sequences the registry behind it, so a
  cold cluster races. The compose service runs `restart: on-failure:5` -- a budget a fast
  crash-loop burns through in seconds, leaving the registry down permanently and reading as "the
  agents have no tools". So `KvError` is retried with bounded exponential backoff and raised once
  the budget is spent; `KvConfigMismatch` propagates on the first attempt, because config drift
  does not heal. `retry_with_backoff` is deliberately NOT used: it never raises, so it would
  downgrade the mismatch to a log line -- the exact swallow the distinct exception type exists to
  prevent.

- **The heartbeat registry's wiring was extracted to `build_heartbeat_collection_registry`.**
  It sat inline in `RegistryServer._start_handlers`, which cannot be driven without standing up
  the whole serve loop, so the scope decision would have been covered by the static walker alone.
  The extraction mirrors the reason `apply_rbac_factory` is already a method -- the file's own
  comment says "extracted to a method so tests can drive the same code path without binding to
  private state" -- and gives the process's two registries symmetric, testable builders.

- **Ledger bug 21 is fixed at the resolver, and the hub must drop its compensating entry.**
  `_agent_router` now declares `{ns}_agent_config`, the unprefixed bucket
  `agent_router/proxy.py` binds directly. It is declared **read-only** (`writable=False`):
  CLAUDE.md's Config Source-of-Truth makes `platform.agents` the source and this bucket a hot
  cache written only by the hub's admin endpoints, the router reads it at exactly one site, and a
  KV read is a `$JS.API` request rather than a `$KV.` publish -- so the write grant costs nothing
  to withhold. **The hub's `static_nats_grants._EXTRA_RESOURCES["agent_router"]` carries the same
  bucket (at `writable=True`) with "reported upstream" written beside it, and must be deleted
  when this lands**, or the generated conf declares the bucket twice and the pinned conf test
  fails. That deletion is a hub edit this shard may not make.

- **W3T-02 is met as far as an infra principal admits, and the residual is stated.** The scope
  input is the `Principal` member, not a configurable value: `THREETEARS_NATS_USER` is
  deliberately NOT read to derive it, because an operator-settable env var deriving the key scope
  would let a typo silently re-key the whole cache. What binds the two is that the process
  connects as the `registry` static user; a process connecting as anything else writes keys its
  own grant does not cover and is refused at the broker. For a POD principal the same guarantee
  needs the key-side id sourced from the authenticated identity -- see the pinned-pair test's
  docstring, which states that residual rather than letting the green test imply it is closed.

- **Bug 20 was NOT taken.** The `{ns}_channels_deliver` misspelling in `_agent_pod`, `_hub` and
  `_channel_adapter` is the same shape as 21 and lives in the same file, but it is a separate
  ledger entry that this shard does not own, and correcting it has the same cross-repo coupling
  as 21 (the hub's `_DROPPED_RESOURCES["slack"]` would go stale, and its enforcement test
  "refuses a drop whose name the resolver no longer emits" -- so fixing it upstream would fail the
  hub build rather than merely duplicating a grant). It stays unowned.
