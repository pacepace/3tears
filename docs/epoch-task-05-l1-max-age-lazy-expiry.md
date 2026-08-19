# epoch-task-05: Bound L1 staleness with lazy max-age expiry

**Status:** BUILT, not shipped -- no PR, not merged, not released. Landed as a series of commits on the branch that carries this file; `git log --oneline
-- packages/core/src/threetears/core/cache` is the current answer, and a range written here
goes stale the next time any of it is touched. Reshaped
after review, which found that the naive version is data loss on two shipped packages, and
again after the cumulative review, which found expiry made silent write loss routine on the
read paths that do not repair.
**Scope:** `3tears-core` (`cache/base.py`, `cache/sqlite.py`, `collections/registry.py`,
`collections/scan_cache.py`, `data/collection_factory.py`). Behaviour also changes for
`3tears-channels`, `3tears-registry`, `3tears-geo` and `3tears-agent-tools`, which read L1
directly.
**Depends on:** nothing. Independent of epoch-task-01 through 04.

---

## Objective

Give L1 row entries a maximum age, enforced lazily on read, so a stale entry self-corrects
within a bounded window whatever caused it.

## Why the existing L2 TTL does not already do this

`packages/core/src/threetears/core/cache/kv.py:161` already sets the `collections` bucket
to a 7200s TTL, so every L2 entry self-expires. That bounds nothing here: a pod holding an
L1 hit never consults L2 at all (`collections/base.py:956-963`). The residue this shard
exists for lives entirely in L1.

epoch-task-04 investigated a publisher-side queue for that residue and concluded it should
not be built, because nats-py already buffers and replays publishes across a reconnect. It
named two gaps that survive, and this shard is the only mechanism in the series that
touches either: an invalidation dropped when the outbound buffer overflows, and a pod whose
*subscription* is partitioned while its peers stay healthy, which misses every invalidation
published in that window and never knows. No publisher-side mechanism can reach the second
one. A max-age bounds both.

## What the bound actually bounds, which is not what it first looks like

**Expiring an L1 row resolves against L2, not L3.** A pull-through consults L2 first and
stops there on a hit, so the freshest value this mechanism can produce is whatever L2
holds. That is not a limitation, it is precisely the fit: the residue being bounded is a
peer's write whose invalidation was lost, and that write *did* reach L2, because
`_save_to_l2` is on the same path as the L3 write. Only this pod's L1 copy is stale.

Worth stating because it is easy to reason about this as "L1 expiry re-reads the database"
and then stage a test that way. It does not, and such a test fails for the right reason.
The bound is on how long one pod may disagree with the shared tier, not on how long the
shared tier may disagree with durable storage. L2 has its own 7200s TTL (`kv.py:161`) for
the latter.

## The exclusion that makes this safe, and is not optional

**Expiry must never apply to a collection with no L3.** Two shipped families set
`self.l3_pool = None` and raise from every L3 method:

- `packages/channels/src/threetears/channels/presence/collection.py:119` (pool),
  `:188-200` (`fetch_from_store` raises), with the rationale at
  `presence/l1_cache.py:13-18`: "presence is transient by construction".
- `packages/registry/src/threetears/registry/heartbeat_collection.py:115`, `:235-249`.

Both override `get()` to consult L1 then L2 and return `None` on total miss, so an expired
entry does not raise. It silently becomes "this row does not exist". In the L1-only mode
that `collections/base.py:1180-1184` documents as supported, a CAS mutate then sees an
absent room, and `save_entity(self.create(...))` writes a fresh one-member room over a
room that had ten. Membership destroyed by a timer, no error anywhere.

So the mechanism is: **expiry is off unless the collection has an L3 pool, and off by
default even then.** Keyed off `self.l3_pool is not None`, not off good intentions. This
is a correctness precondition, not the performance nicety an earlier draft called it.

## Where the stamp is written (the invariant, stated once)

**The stamp records when the row was last obtained from L2 or L3, not when it was last
touched in L1.** That is the whole point: a locally-authored write does not make a row
fresher with respect to what other pods know.

Consequences to implement deliberately:

- Stamp in the pull-through path (`collections/base.py:737-738`, `:742-743`).
- **Stamp in `reload_entity` too.** It is not a pull-through, but it fetches from L3 and
  writes the result into L1, so the row's provenance is a lower tier and the invariant
  applies. Easy to miss because it is reached from `BaseEntity.reload()` rather than from
  `get`/`ensure`; found in review of Chunk 02, not in the original design.
- Do **not** stamp in `upsert`. If it stamped there, `set_field_sync`
  (`collections/base.py:461-465`), `__setitem__` (`:800-805`) and
  `packages/agent/tools/src/threetears/agent/tools/collections.py:353-360` would each
  renew a stale row's lifetime by touching one field, making exactly the residue this
  shard targets immortal.
- A row with no stamp (locally authored, never pulled through) is **treated as fresh**.
  It reflects a write this pod made, and expiring it would revert a local write.
- **`upsert` preserves an existing stamp when the caller supplies none.** Discovered while
  building Chunk 02, and load-bearing:
  `packages/agent/tools/src/threetears/agent/tools/collections.py:353-361` reads a row from
  L1, mutates one field, and upserts it straight back. Since reads strip the stamp (below),
  that write-back carries none. Without this rule it would silently clear the stamp, and a
  row that had been pulled through hours ago would start reading as locally-authored and
  therefore never expire. Absence of the key means "unchanged", not "clear it".

## Injection points (the earlier draft named the wrong one)

Pointing only at the DDL generator would have produced a column that exists in the table
and not in the schema registry, and `upsert` filters writes to the registry
(`cache/sqlite.py:277-280`). The stamp would be silently discarded on every write, the
column always NULL, and expiry a no-op with every DDL-checking test passing.

The full set:

| What | Where |
|---|---|
| DDL | `cache/sqlite.py:524-547` |
| Schema registry | `cache/sqlite.py:139` |
| Strip from results | `_deserialize_row`, the single funnel both `select_by_id` and `select_batch` already pass every row through |
| Dynamic collections | `data/collection_factory.py:129,243` builds its own metadata and calls `initialize` per table |

`execute_query` is deliberately exempt: it returns raw dicts against explicit column lists
and is used only for non-entity tables (`collections/flush.py:207`,
`geo/features.py:163,194`), as are the raw-connection reads at `geo/features.py:127-137`.

`build_select_clause` returns literal `"*"` when `columns is None` (`cache/base.py:42-43`),
which is every production call, so the stamp does come back and must be stripped. Stripping
in `_deserialize_row` covers the projected case for free: a caller naming the stamp
explicitly still gets it removed, so no projection has to be rewritten and no caller can
name its way past the boundary.

**Consequence for Chunk 03, decided here rather than discovered there.** Because reads strip
the stamp, the expiry comparison cannot live above the backend: nothing above it can see the
value. So the backend owns the *mechanism* (compare, and delete the expired row), and the
collections layer owns the *policy* (whether this collection expires at all, and at what age)
by configuring the backend per table. That split is the reason the exclusion below is
expressible: `l3_pool is None` is a collections-level fact, and a collection that knows it
has no L3 simply never configures a max age.

## Column name

Not `stored_at_monotonic`. `collections/scan_cache.py:53` already declares that column on
`collection_scan_cache`, initialised on the pod's shared L1 backend at `:115-116`. A
blanket injection by that name emits a duplicate column and fails DDL at startup; a
blanket strip by that name breaks `ScanCache.get`, which reads it at `:132`.

Use a reserved name (`_3t_cached_at`), declare it reserved so a user table using it is an
error, and exempt the tables that are not entity caches: `write_buffer`
(`collections/flush.py:30-39`) and `collection_scan_cache`.

Geo's R-tree tables need no exemption entry, though an earlier draft of this section said
they did. They are created by raw `CREATE TABLE` / `CREATE VIRTUAL TABLE` against the
connection (`geo/features.py:127-137`) and never pass through `initialize()`, so the
injection cannot reach them. Adding them to the exemption list would be inert, and inert
entries are worse than absent ones: the next reader takes the list as the complete account
of what is exempt and why.

Type is `Float` (`REAL` under `sqlite.py:581-582`), not `Text`. `ScanCache` stores its
stamp as a string and parses it back per read (`scan_cache.py:171`, `:132`); copying that
would make "one integer comparison per read" false.

## Reuse: extract, do not mirror

`collections/scan_cache.py:128-139` is the same predicate. Mirroring it leaves two
implementations of one five-line rule with the same subtle parts (injected clock, disposal
of the expired entry). Extract one helper beside `build_select_clause` in
`cache/base.py:18-51`, which exists for precisely this reason ("Shared by every backend so
projection validation cannot drift between them"), and have `ScanCache.get` call it.

Note `scan_cache.py:135-138` deletes the expired entry rather than leaving it, with the
reason given: "drop it now rather than leave a tombstone that every later read has to
re-evaluate." Treating expiry as a miss without deleting is fine while pull-through
re-upserts, but when the row is genuinely gone from L3 the pull-through returns `None`,
the entry is never deleted, and every read re-evaluates it forever. Delete on expiry.

## DuckDB is mostly out of scope, and the exception is instructive

An earlier draft required both backends to change together. `DuckDBBackend` has **zero**
production construction sites (all are tests), and it is not a peer today: `initialize`
short-circuits on re-entry, so the dynamic-collection path's per-table lazy init silently
no-ops after the first table, and `upsert` uses `INSERT OR REPLACE`, which NULLs unlisted
columns. An "identical behaviour" acceptance criterion cannot pass without fixing all of
that.

**Two DuckDB changes were forced anyway, and the reason generalises.** The stamp is written
by the collections layer, which is backend-agnostic, so scoping a backend out of the READ
side does not scope it out of the WRITE side. `upsert` had to start filtering to the
registered schema, as SQLite always did, or the stamp reaches SQL against a table that
declares no such column and every pull-through fails. Separately, the read side refuses a
bound with `NotImplementedError` rather than accepting one it cannot honour.

So: expiry itself is SQLite-only, but "out of scope" could not mean "untouched". Bringing
the rest of DuckDB to parity, or deleting it, is still separate work.

## Why monotonic, and the constraint that keeps it true

`time.monotonic()` is process-wide in CPython, so the thread-local connection pools
(`sqlite.py:170-176`) are fine, which is the first objection worth answering. It cannot
step backwards under NTP correction the way a wall clock can, and it is never compared
across machines.

**L1 storage must remain process-local for this to hold.** It is today: SQLite L1 uses
`file:/{db_name}?vfs=memdb` (`sqlite.py:106-112`). If L1 ever becomes file-backed and
outlives the process, a monotonic stamp written by one process and read by another is
meaningless, and this mechanism must switch to a wall clock or be disabled.

## Configuration: per-collection, defaulting to 3600s

**The design decision is the shape of the knob, not its value.** Per-collection
configuration lives on `CollectionRegistry`; a single global value would put it on the
backend constructor instead.

**It must NOT share `_overrides` with the tier overrides**, though an earlier version of
this section said it should. `register()` hard-resets that dict whenever it is handed any
tier kwarg, and wiring commonly configures before registering, so a bound stored there is
silently dropped: expiry off while the operator believes it on. It lives in its own
`_l1_max_ages` dict, with a regression test named for exactly that. Anyone tidying two
table-keyed dicts into one will reintroduce the defect, which is why this is recorded
rather than quietly fixed. Per-collection is the choice, because collections in
this repo differ by orders of magnitude in read volume and in how much staleness they can
tolerate, and a global value would be tuned for the worst of them.

The default is **3600s**, and it is tuning rather than architecture: it can move without
touching anything structural, which is the point of settling the knob first. The reasoning
for that starting value is thin and should be treated as such. The L2 bucket TTL is 7200s
(`kv.py:161`), so a default below it means a refetch usually resolves at L2 rather than L3,
and a default above it is largely inert. 3600s sits under that with room.

Expiry is off by default regardless, per the exclusion above. A collection opts in --
and 3600s is what it gets if it opts in without naming a number
(`DEFAULT_L1_MAX_AGE_SECONDS`). Those two statements only look contradictory: there is
no fleet-wide default, and there IS a default for the value, which is the only place a
default can live once nothing is bounded until asked.

## Interaction with the deferred write buffer

`collections/flush.py:195-199` documents the ordering: persist to L3 first, evict from L1
only after the durable write is acked. Expiry evicts on a timer that knows nothing about
the buffer, so a row written under a deferred flush strategy and expired before its buffer
entry drains would pull through and serve the pre-write value.

The "no stamp means fresh" rule covers this **only for a row this process authored**. An
earlier version of this section claimed it covered the deferred case outright; it does not.
A row pulled through earlier keeps its stamp across a local save, because `upsert`
preserves a stamp the caller does not supply, so it can age out while its write is still
buffered.

That is survivable rather than safe-by-construction: the pull-through reads L2, which the
same save already wrote, so the new value comes back and nothing reverts. Without L2 wired
it reads L3 and serves the pre-write value. The code comment at `_entry_is_fresh` states
the same limit, because one rule being load-bearing for another is not obvious from either.

## Acceptance

- A collection with `l3_pool is None` never expires an entry, proven against
  `_L1L2OnlyCollection` and `HeartbeatCollection` directly.
- A read past max age misses, pulls through, and **deletes** the expired entry, proven
  with an injected clock rather than a sleep.
- `select_batch` applies the same predicate as `select_by_id`.
- The stamp never appears in a row returned to a caller, including under projection.
- `ScanCache` uses the extracted helper; there is one implementation of the predicate.
- `collection_scan_cache` and `write_buffer` gain no injected column.
- A row written locally and never pulled through does not expire.
