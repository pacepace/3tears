# Scoped snapshot: whole tables in L2

`threetears.core.collections.scoped_snapshot.ScopedSnapshot` keeps tables that a pod needs whole
in NATS (the L2 tier), and loads them into a DuckDB L1. It also covers the pod-owned NATS pieces
under it: `threetears.nats.object_store`, `NatsKvBucket.watch_prefix`, a tool pod's Object Store
grant, and `threetears.nats.object_store_requests`.

## Decision

**Pace chose NATS L2 over S3 or parquet files (2026-10-07).** The question came from the ENR tool
pod. A starting replica waited 22 minutes reading its report tables from L3, a thousand rows a
statement. After every refresh it spent about 5 minutes rebuilding its whole DuckDB copy. The first
proposal was a parquet file in object storage. Pace asked instead how 3tears' L2 could hold the
copy properly.

A probe on the local stack (nats-server 2.12.6) settled the shape:

- One epoch of the candidates table plus its geography, split by state, is **16.9 MB** as Arrow
  IPC with zstd. Per-state chunks run from 7.8 KB to 1.69 MB, so all six tables are likely under
  50 MB. That fits the local NATS and cobalt's 1 GiB memory store per server with no setting
  changed.
- Loading 102 chunks into DuckDB took **0.57 to 0.59 s**.
- Swapping in one state took **62 to 83 ms** end to end. About 10,000 reader queries ran during the
  swaps, and none saw a torn read.

S3 or parquet stays a fallback only. L3 is the truth, and NATS is a cache rebuilt from it.

## Shape

- **One chunk per scope and table.** Each is one Arrow IPC stream, zstd-compressed, in the pod's
  own Object Store. It is named `{snapshot}/{scope}/{epoch}/{table}.{column digest}`. The digest
  means a rebuild under other columns never reuses an old chunk's name. A table declared
  append-only holds a base chunk and the pieces after it instead (below).
- **One pointer per scope.** It is a small KV entry naming the scope's epoch and its chunk for
  every table, so a scope's change across tables is one swap.
- **An index.** It names every scope, so a missing pointer is noticed, not skipped.
- **A rebuild claim.** It is a `KVLease` hold on `{name}.rebuild` in the pointer bucket, with a
  NATS per-key TTL, renewed while the holder works and released only while still its own. It is
  contended, so a lost one is never taken back: the rebuild that lost it says so and leaves the
  store's sweep to a rebuild that held its claim throughout.
- **Compare-and-set on pointers and the index.** A pointer never moves to a lower epoch. The
  index merges what racing writers added. A replica's L1 never goes back to a lower epoch of a
  scope.
- **Every row has a scope.** A publish with no scope is refused. A rebuild that finds rows with a
  null scope fails and names the table, instead of serving a copy that claims to be whole and
  isn't.
- **Retirement.** Superseded chunks are deleted by the bucket's declarer (the hub), in batches of
  at most `MAX_RETIRED_OBJECTS`. A pod holds no purge. A reader that loses a race to a retired
  chunk reads the scope's pointer again.

## A writer that stages

**Decided 2026-10-07, for the ENR pod's refresh.** A first load writes every state of 517,000 rows;
publishing after the commit with the rows in hand would hold them all in memory. So a writer stages:

- **Stage, then publish.** As each scope is written to L3, `stage()` writes its chunks under the
  write's version, which the commit makes the scope's epoch. After the commit, `publish_staged()`
  moves every staged pointer. Nothing reads a chunk until its pointer moves, so a write that never
  commits shows nothing.
- **Carry only from the epoch the writer saw.** A table a write did not change in a scope keeps the
  scope's chunk, but only when the pointer is at the epoch the writer read before it began. Any
  other pointer may hold a dead write's rows that L3 has and no chunk does. The move is a
  compare-and-set against exactly the entry the carry was judged on.
- **Anything else goes to the catch-up.** A skipped scope's L3 epoch is now ahead of its pointer, so
  `catch_up_from_l3` republishes it from L3, the slow path but the correct one. A writer that finds
  an earlier write unfinished stages nothing and leaves every state to the catch-up.
- **Hold the claim across commit and publish.** Otherwise a replica waiting on the write rebuilds
  every scope from L3 in the moment between them.

## Append-only tables: a base and its pieces

**Decided 2026-10-19, for the ENR pod's LSF projections** (about 1,000 rows a model run, a run
every 3 to 5 minutes across nearly every state, about 400,000 rows by morning). Republishing a
scope's whole rows to add a run's few means every replica fetches and reloads the whole scope each
time. A table declared append-only (`SnapshotTable(append_only=True)`: rows are only ever added,
and a row once held never changes or goes away) is published by what each write ADDED.

- **A base and its pieces.** For such a table a scope's pointer names a base chunk and an ordered
  list of pieces after it, each the rows one write added, named by scope, epoch and column digest
  as chunks are: `{snapshot}.p/{scope}/{epoch}/{table}.{column digest}`.
- **A writer gives the added rows, apart.** `stage(..., added=)` and `publish(..., added=, after=)`
  take an append-only table's added rows in their own argument, so the call says which meaning
  it has; `rows=` keeps meaning every row (an append-only table given every row gets a fresh base).
  Added rows whose keys this replica already holds for the scope are refused before anything is
  written, so a caller handing over every row as added rows is told, never served a repeated piece.
- **A piece follows exactly the chain it was computed after.** The piece is appended to the chain
  the scope's pointer holds at the epoch the writer saw before its change (`carry_at` for a stage,
  `after` for a publish), by a compare-and-set against exactly that entry. This is what keeps every
  pointer's chain equal to L3's rows at its epoch: two writes committed in one order and published
  in the other can never put the later one's rows on a chain missing the earlier one's. A publish
  whose pointer is not at `after` on the chain this L1 holds, or that loses the compare-and-set,
  moves nothing and changes nothing here (this L1 takes a publish only once its pointer has moved,
  so `applied_epoch` and the last change name only epochs the pointer took). `publish` returns
  `False`, and the scope is shown behind in `status()` and rebuilt from L3 at the next pass, which
  holds the change. A skipped stage leaves the scope with its pointer behind L3's epoch, which
  `catch_up_from_l3` republishes. A new scope's first piece is its base.
- **Other tables of the snapshot.** A publish of whole rows only moves the pointer exactly as a
  snapshot with no append-only table does: the compare-and-set retried against racing writers.
  A publish that carries added rows moves the scope's one pointer from exactly the entry at
  `after`, and its whole tables move with it, since one pointer names every table of the scope; a
  writer that wants a whole table's change published regardless of a race publishes it whole,
  apart from its added rows.
- **A replica appends what follows what it holds.** When what it holds of a table is a leading run
  of the new chain (the same base, its pieces the first of the new list), it fetches only the
  pieces after them and appends them, in the same one-transaction swap readers see today. Anything
  else (a new base, a chain it does not hold) reloads the table's scope whole from base and pieces.
  The judgement is by names alone, so a lost or out-of-order pointer event converges: the next
  pointer seen is compared with what is held, and every piece after it is fetched. A cold replica
  loads base and every piece.
- **One publish sequence.** A publish, a rebuild and a fold all write through one sequence: a
  write claim on the epoch, each table's chunk written whole, appended as a piece or carried,
  every named chunk checked to be there, then the pointer moved.
- **Compaction.** Past a threshold (`compact_pieces` pieces, or `compact_bytes` of them, per table
  per scope) the writer that moved the scope folds base and pieces into a new base at the pointer's
  epoch, under a write claim, by compare-and-set; the folded pieces go by the sweeper's one rule.
  The new base is the old base and its pieces concatenated in order, not re-sorted, so a replica
  that loads it holds its rows in the same order as one that appended them. A replica already
  holding the scope at that epoch adopts the folded pointer without fetching it: the rows of an
  append-only table at one epoch are one set. A rebuild and a catch-up write a fresh base with no
  pieces. `status().appended` says, per append-only table, how many pieces each scope holds and when
  its base was last formed.
- **Everything else holds.** L3 is the truth; the pointer compare-and-set, the write claims, the
  sweeper and the status cover pieces; a table not declared append-only keeps one whole chunk per
  scope; `carry_at` carries an unchanged table with its pieces.

### The pointer format, and two code versions

The pointer is a persisted format read by every replica of every code version in a rolling deploy,
so it is a lock-in decision. It stays one KV entry per scope at the same key, and is written so that
a replica that predates pieces (0.72) can never misread it and a replica that knows them (0.73)
reads every older pointer:

| Field | What it holds |
|---|---|
| `tables[t].object`, `tables[t].rows` | the table's base chunk and its rows: exactly the 0.72 fields |
| `tables[t].pieces` | the ordered pieces, each `{object, rows, epoch, bytes}`; present only when there are some |
| `tables[t].compacted_at` | when an append-only table's base was last formed whole (epoch seconds); optional |
| `schema[t]` | the table's column digest, followed by `+pieces` exactly when `pieces` is present |

- **0.72 fails closed.** It compares each table's schema entry with its own column digest before it
  loads anything. A marked entry never equals it, so 0.72 treats the scope as another code
  version's: it serves the scope from its own L1 rebuilt from L3, and it does not repoint the scope
  at the same epoch (the rule in "Two code versions at once" below). It never decodes a piece as
  rows. Nothing had to change in 0.72 for this.
- **0.72 cannot sweep a piece.** It lists and judges only objects under `{snapshot}/`; pieces live
  under `{snapshot}.p/`. A snapshot's name holds no `.`, so no other snapshot's chunks share that
  prefix. 0.72 keeps the base, because the base is in the field it reads.
- **0.73 reads 0.72.** A pointer with no `pieces` is a table of whole chunks: a base with none.
- **Back to the common shape.** A pointer whose tables have no pieces is written exactly in the 0.72
  shape, so a rebuild, a catch-up or a compaction puts every replica, 0.72 included, back on
  pointers it loads.
- **A future format fails closed too.** 0.73 refuses an entry whose pieces and mark disagree, and
  keeps any other suffix on the digest, so a pointer of a later format reads as other columns.

The alternatives were a second pointer key per scope (two pointers a scope could disagree across,
and 0.72 would see the scope's pointer missing and rebuild it under the old key, the two versions
fighting) and a format number 0.72 does not read (it would load the base alone and serve a scope
missing every piece: the misread this rules out).

## Bounded memory: a rebuild and a load hold a few scopes at a time

**Decided 2026-10-22, for the reports tool pod.** Its DuckDB copy is about 360 MB and its snapshot in
NATS about 300 MB, but a rebuild from L3 read every scope's rows as Python dicts before it published
any: 0.9 to 2 KB a row, at least 18 GB for 16.3 million L3 rows. Pods were OOMKilled at 8 GiB minutes
into the rebuild, and most of what they held the copy's retention dropped afterwards.

- **Scope by scope, a page at a time.** A rebuild reads each scope from L3 a keyset page at a time
  (`complete_copy.l3_row_pages`). Each page goes through the backend's `export_rows` as it arrives,
  so it is held as the L1 keeps it: typed in Arrow, and cut by a backend's retention. The scope is
  then published like any whole publish (this L1, its chunks, its pointer) and let go. At most
  `l3_concurrency` scopes (default 4) are held at once, read beside the one being published, and
  published in scope order.
- **The writer's stamp, per scope.** Each scope is published only if the writer's stamp, read after
  its rows, is still the one taken before the epochs were read. Once a write has committed, nothing
  more is published and the attempt reads again. A scope already published at the epoch L3 still
  records is not read again.
- **Partial is never complete.** Scopes publish one at a time, as the publish loop always did. The
  index is written whole, and the copy declared ready, only after every scope is in. A rebuild that
  dies part way leaves each scope it published whole at its epoch, and no index naming the snapshot
  complete.
- **A load from L2 the same way.** A load fetches up to `l2_concurrency` scopes' chunks at once and
  commits each scope in its own transaction, in scope order, as soon as it and the scopes before it
  are in. The copy is ready only once every scope is in.
- **`export_rows` costs what its rows cost.** `DuckDBBackend.export_rows` types the rows beside the
  table, cast to its columns, and never in it. It used to delete the scope's held rows in a
  rolled-back transaction, which made a page cost the whole scope.

Measured on a synthetic 3 million rows in 50 scopes (12 columns, Linux, an L3 that answers at once):
the rebuild's peak went from 4.2 GB to 1.6 GB (1.2 GB with `l3_concurrency=1`), and the load's from
2.4 GB to 1.7 to 1.9 GB. What remains is the copy itself and a few scopes. With 400 scopes the rebuild's
went from 3.7 GB to 1.1 GB: what a rebuild holds beyond the copy tracks its largest scope, not the total.
The price is time where L3 and NATS cost nothing: the rebuild took 24 to 31 s against 22 s (each page
typed on its own), the load 3 to 5 s against 2 s (a transaction per scope).

**An export is sized for its rows (2026-10-23).** On dev the rebuild still peaked at 7.5 GiB and the pod
stayed there after it was ready. Two causes, both in DuckDB's allocations, which neither Python nor Arrow's
pool counts:

- DuckDB sizes an Arrow export's buffers for a million rows unless asked for fewer, and a held export
  keeps them. A 999-row page held 2.2 MiB resident for 0.23 MiB of data, so a scope's pages held ten
  times the scope. Every export now asks for batches of at most its rows (`fetch_arrow`): the rows
  given to `export_rows`, the scope's rows counted by `export_partition`. With DuckDB 1.5.5 and pyarrow
  25.0.1 (Linux arm64, 1 million rows in 50 scopes) the rebuild's peak went from 8.4 GB to 0.9 GB; with
  3 million rows it peaks at 1.7 GB, where before it ran out of a 10 GB container.
- On Linux x86_64 DuckDB allocates through its own jemalloc, whose arenas give freed pages back to the
  OS only when their thread allocates again. A pod that sits idle after its rebuild keeps them. Every
  backend opens its database with `allocator_background_threads`, so a thread of jemalloc's own gives
  them back. Measured with jemalloc preloaded for the whole process (arm64, where DuckDB has none of its
  own), 1 million rows on the old export: 1.27 GB stayed resident without the thread, 0.85 GB with it.

## Chunk lifecycle

One rule (`_Sweeper.deletable`, beside the snapshot in its module) decides whether a chunk may be
deleted, and every deletion path goes through it. It is judged by state, never by age, against the
pointers and the write claims as KV holds them when the sweep runs (read after the chunks are
listed), never a replica's view, which lags another replica's publish:

| The chunk | Kept or deleted |
|---|---|
| Named by its scope's pointer, as a base or a piece | Kept: it is being served |
| At an epoch a live write claims | Kept: it is being written, or waits for its pointer to move |
| Anything else | Deleted: it serves nothing and nothing will point at it |

- **A write claim** is a key `{name}.w.{epoch}.{writer}` in the pointer bucket, one per write
  (`writer` names the replica and the write), taken before the write's first chunk at that epoch
  and held until a pointer names what it wrote (a staged write: until `publish_staged`, the claim
  riding on its `StagedScope`; a publish or rebuild: until its pointer moved). Each is a `KVLease`
  hold on the pointer bucket it was handed, written with a NATS per-key TTL and renewed at a third of
  it while the writer lives; a writer that dies stops renewing and its KEY lapses, so it stops
  protecting what the dead write staged and stops holding off the hub's orphan purge, which reads
  only whether a claim key exists. A claim found gone while its writer lives (lost with its bucket,
  or lapsed through an outage) is taken again at the next renewal, and one of its own renewals that
  landed unseen is adopted, not taken for another holder's; a rebuild that follows a lost pointer
  bucket sweeps nothing, since the claims went with it. A claim truly lost (another holder on its
  key) stops its write: no further chunk is written under it and no pointer moved onto what it
  wrote, the publish fails, and a staged scope is left to the catch-up.
- **Older chunks go as each pointer moves**, on every path (publish, staged publish, rebuild,
  catch-up), so a run that fails part way leaves nothing superseded behind for the scopes it moved.
- **A write that will not commit gives its stages back** (`discard_staged`: its claims released,
  its chunks retired by the rule), and a writer taking over a dead one may name its epoch
  (`discard_epoch`). `publish_staged` lets its stages' claims go on every way out, raised or
  returned. A stage a LIVE writer drops without either keeps its claim renewed until the snapshot
  stops (`stop` ends every claim the replica handed out and nobody released); a DEAD writer's
  claims lapse with their per-key TTL, and its stages go at the next sweep after that.
- **A full store** (`ObjectStoreFullError` on a chunk write) is swept by the rule and the write tried
  once more. It recovers when what fills it is chunks no pointer serves and no live write claims; a
  store full of what is served needs a larger bound.
- **No pointer moves onto a missing chunk.** A staged publish checks every chunk it names first,
  and a publish that found its chunk already written checks it before moving; a missing one leaves
  the scope to the catch-up, which republishes it from L3.

The hub's own sweep of chunk subjects no object names (pieces of a put that failed part way, since
an object's chunks land before its metadata) runs at each retire only while no write claim stands
in the pod's pointer bucket: while one does, an unnamed chunk may be a put in flight. Judged by
state, as the chunks are, never by age.

## Two code versions at once

A rolling deploy of a column change runs replicas with different column sets over the same
pointers. Each pointer names the columns its chunks hold (`schema`, per-table digests).

- **A replica loads only chunks of its own columns.** One check (`_loadable`) sits in the fetch
  every load passes through, so no path can put another version's chunks in its L1.
- **It does not repoint the other version's scopes.** A rebuild moves a scope's pointer only
  forward: a same-epoch pointer of other columns stays until a write moves the scope to a later
  epoch. Meanwhile the replica serves the scope from its own L1, rebuilt from L3, and its status
  says how many scopes that is. Without this rule each version would
  repoint the other's scopes on every rebuild, each round the slow path.
- **A write moves the pointer to the writer's columns.** A later epoch always moves it. A replica
  of the other columns cannot load it, so it rebuilds that scope from L3 and serves it.

## The pod's own buckets

A tool pod gets an Object Store and a pointer bucket only when its registry row opts in. The hub
declares both when the pod asks. Both are bounded: the Object Store by `max_bytes` with
`discard: new`, and the pointer bucket by `max_bytes` too. Neither allows rollup headers, because a
rollup is a purge done by publishing. The hub deletes both when the pod is removed.

The pod's grant (`JsCapability.OBJECT_STORE_OBJECTS`) allows:

- `STREAM.INFO` (bind)
- a direct get of metadata
- a named-consumer create filtered inside the bucket
- the `$O.` publish

It never carries a stream-management verb, `STREAM.MSG.GET` or `STREAM.MSG.DELETE`.

## Known residual: a pod can put its own bytes on any subject

**This is a platform hole. No grant closes it.** It is pinned by
`packages/nats/tests/integration/test_a_pod_can_bounce_its_bytes_to_any_subject_residual_live.py`.
That test asserts the hole exists, so when it starts failing, the hole has closed: remove the test
and this section.

Probed on 2026-10-07 against nats-server 2.12.6, with a real minted tool-pod JWT applied as config
permissions:

1. **Push-consumer delivery is not checked.** A named push consumer on the pod's own bucket is
   admitted by the grant, because its filter rides in the create subject. Its `deliver_subject`,
   though, rides in the request body, which no subject permission sees. The server delivered the
   pod's stored bytes to `bounce.tools.internal.<another pod>` and to another principal's inbox,
   although the pod may publish to neither.
2. **Reply subjects are not checked.** Every request the grant admits is answered on the request's
   reply subject, and NATS never checks that subject against the requester's permissions. A direct
   get of a value the pod wrote came back verbatim on a subject the pod may not publish. The same
   holds for pull `MSG.NEXT` and any JetStream API call.

So changing the read shape (pull consumers, ordered consumers, direct gets) closes nothing. Any pod
that can write a value and read it back has this, through every L2 grant:

- its own KV buckets and Object Store
- its owner keys in the shared buckets
- its scope of the collections bucket

**The defence belongs to the receiver.** If a subject's receiver trusts the subject alone (no
signed token, no correlation id it minted itself), a pod can feed it the pod's own bytes. Pace
decided on 2026-10-07 to audit those receivers; their fixes are separate work. The other route is
separate NATS accounts per principal with explicit exports and imports, since deliveries and
replies stay inside an account.
