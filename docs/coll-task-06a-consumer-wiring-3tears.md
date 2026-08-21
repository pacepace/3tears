# coll-task-06a: Wire the Scope — 3tears

## Objective

`coll-task-03` refuses an L2 client with no scope. Supply it at the 3tears
consumer, and add the enforcement rule the other two repos' shards rely on.

Split by repo because the three land at different points in the version dance
(`coll-sequence.md`, Landing mechanics) and because a single shard spanning
~40 sites in three repos is not one context window.

Site list and ratified decisions: the evidence ledger, Part 2b and Part 4.

---

## The sites — two registries in one process

The registry server builds **two** `CollectionRegistry` instances, and an earlier
draft of this shard saw only the first:

- `registry/server.py` — configures `l1_backend` + `l2_client`, starts the
  invalidation listener, and holds `HeartbeatCollection`.
- `rbac_stack.py` — a second registry, `configure(l1_backend, l3_pool)` with no
  `l2_client`, then five ACL collections each passing `nats_client=` directly.
  Wired from `server.py`. Exact mirror of the hub's `gateway/acl.py` shape.

Both need a scope; both resolve to the registry principal.

`core/data/collection_factory.py` and `core/data/store.py` are library helpers
and own no scope; a process entry point does.

`HeartbeatCollection` is **`L3 = None`** and lives in the collections bucket, so
this process's L2 is a source of truth, not a cache. Coordinate with
`coll-task-05a`, which adds the collections grant the registry has never held.

**The registry currently runs its L2 collection against a bucket it is not
granted** — see `coll-task-05a`. Wiring a scope here without that grant lands a
correctly-scoped key on a bucket the principal cannot reach.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| W3T-01 | The registry server sets `kv_key_scope` from `kv_key_scope_for(Principal.REGISTRY)` | P0 |
| W3T-02 | The scope input is the authenticated identity the process presents at connect, not local config | P0 |
| W3T-03 | The enforcement rule below exists and covers all **five** wiring shapes | P0 |
| W3T-04 | Both registries in the registry-server process are scoped — `server.py` and `rbac_stack.py` | P0 |
| W3T-05 | The process opens the collections bucket eagerly at startup via `ensure_kv_bucket`, before `configure(l2_client=)`, so a `KvConfigMismatch` raises at wiring rather than in a request path (`coll-task-04a` KVC-05) | P0 |

---

## The drift the enforcement rule must catch

The key-side scope and the grant-side scope are the same string produced in two
places. If they diverge, every read misses silently, falls through to L3, and
nothing logs it: correct answers, a dead cache.

A pinned-pair test proves both sides call `kv_key_scope_for`. It **cannot** prove
they feed it the same input — that residual closes only by sourcing the input
from the authenticated identity. State both halves in the test's docstring so the
next reader does not over-trust it.

The walker must cover **five** shapes: `configure(l2_client=)`,
`register(..., l2_client=)`, `bind_table(...)`, a collection constructor's
`nats_client=` keyword, and a **positional** NATS client — `hub/geo/wiring.py`'s
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

- `packages/registry/src/threetears/registry/server.py` — scope, plus W3T-05's eager open
- `packages/registry/src/threetears/registry/rbac_stack.py` — the second registry
- `packages/nats/tests/enforcement/` — the pinned-pair test and the wiring walker (new files). They belong here, not in the hub: both halves the pair pins (`kv_key_scope_for` in `threetears.nats`, `l2_key` in `threetears.core`) live in this repo.

---

## Anti-patterns

- DO NOT set a scope inside a library factory to make a test pass.
- DO NOT use `conn_id`. Every reconnect orphans the cache.
- DO NOT let the grant side and the key side derive the scope from different inputs.

---

## Success criteria

- [ ] Both registries in the registry-server process set a scope; the process starts and its L2 reads hit
- [ ] The eager bucket open runs before `configure(l2_client=)`
- [ ] The pinned-pair test exists, with its limitation documented
- [ ] The wiring walker covers **five** shapes, including the positional one
- [ ] `./scripts/check-all.sh` and `./scripts/test-integration.sh` green

---

## Verification

```bash
cd 3tears
./scripts/check-all.sh
./scripts/test-integration.sh
```

Live: `nats kv ls` shows the registry's keys under its own scope segment and no
bare `{table}.{pk}` keys reappear.
