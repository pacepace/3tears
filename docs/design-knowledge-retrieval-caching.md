# Knowledge retrieval: stop re-reading L3 on every turn

## The problem, measured

Every agent turn runs **two** knowledge reads — concepts and playbook entries —
and both bypass the Collection they live on:

```python
rows = await self.l3_pool.fetch(sql, *params, customer_scope=customer_scope)
result.append(_row_to_concept_snapshot(dict(row)))
```

`ConceptCollection` is a `SchemaBackedCollection[ConceptEntity]` with an entity
class and the full three-tier stack behind it. `list_visible_to_user` touches
none of it: raw pool fetch, raw dict, hand-built snapshot. No L1 read, no L1
write, no participation in the cache-invalidation broadcast. Same shape in
`PlaybookEntryCollection`.

So on every turn, for every agent, the pod re-reads every visible concept row
and every visible entry row from L3 — over NATS, through the broker, against
distributed Yugabyte — with a **5000 ms** default request timeout
(`nats_proxy.py`, `THREETEARS_NATS_PROXY_TIMEOUT_MS`) and no retry.

Observed on cobalt-dev during a 51-case eval run: `aibots.l3.query: nats:
timeout`, concept retrieval soft-failing, and the turn continuing with NO
governed concepts. Roughly one turn in fourteen. The answer that follows is
indistinguishable from a governed one.

## Why the Collection cache does not already cover it

The by-pk L1 cache cannot express "which rows may this caller see" — the
visibility predicate is a cross-table JOIN against `role_assignments` /
`group_members`. That is a real limitation, correctly documented.

The error is the conclusion drawn from it. "Cannot use the by-pk cache" became
"do not cache at all", so a stable, rarely-changing row set is re-fetched
per turn because the *authorization decision* over it is dynamic.

## The split

Two questions with completely different change rates, currently fused into one
query:

| | changes | cacheable by |
|---|---|---|
| **which ids may this caller see** | on RBAC writes | scan cache, keyed by caller |
| **what are those rows** | on knowledge writes | the existing by-pk L1 |

### 1. Rows through the Collection

Hydrate ids through the Collection's normal by-pk path so rows land in L1 and
are evicted by the `CacheInvalidationMessage` already broadcast on every write.
No new invalidation machinery — this is the mechanism working as designed.

One wrinkle: the concept query `LEFT JOIN`s `datasource_tables` for
`bound_schema_name` / `bound_table_name`, which are not concept columns. Those
resolve by pk against the table catalogue, which is itself cacheable and changes
only on schema import. Resolve them separately rather than re-joining per turn.

### 2. A scan cache on the Collection layer

New, opt-in, general — **not** bespoke to knowledge, and **not** a dict
(see below). A collection declares:

- a **scan key** — here `(user_id, datasource_id, datasource_table_id,
  customer_scope)`
- the **tables the scan depends on** — here `concepts` (or
  `playbook_entries`), `role_assignments`, `group_members`

The registry drops the cached id-set when an invalidation broadcast names any
declared dependency. The RBAC dependency is the load-bearing one: without it a
revoked grant stays visible until TTL, which is a security regression, not a
staleness annoyance. A short TTL backstops anything the broadcast misses.

### Storage

`threetears.core.cache.base.L1Backend` and its duckdb / sqlite / kv backends.
**Never a plain dict.** Two reasons, and the second is the bigger one:

1. Dicts are not async or thread safe; interleaved coroutines lose entries with
   no error.
2. A process-local cache is **per-pod**. The moment another pod writes, every
   other pod serves stale data and nothing says so. Multi-pod caching is the
   reason 3tears exists; hand-rolling local state throws it away.

## Also in this release

- **`retrieve_concepts` must not soft-fail silently**
  (`agent/knowledge/integration.py`). A fault returns `([], [])` and the turn
  proceeds ungoverned. It must return a degraded signal so the caller can
  declare it. The turn need not die — the existing decision was only ever "do
  not break the turn", and that stays true.
- **Delete the SDK's orphaned duplicate.**
  `aibots_agents/runtime/knowledge.py:1240` carries a near-identical
  `retrieve_concepts` with the same soft-fail and **no callers anywhere** in
  `src/` or `tests/`. Fixing one copy and leaving the other is how this bug
  comes back.

## What this does not change

The hub-side `governance_unavailable` imperative stays. It is the signal for
when the cache *and* the fallback both fail, and it is cheap. The difference
between a bad answer and a bad answer that says so.
