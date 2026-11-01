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
  store's sweep to a rebuild that held its claim throughout. Renewal is not progress: a rebuild or
  catch-up that takes no step (a page read, a chunk read or written, a commit, a pointer moved) for
  `rebuild_stall` (10 minutes) lets the claim go and ends with a `TimeoutError`, and a writer's
  `holding_rebuilds` renews it no longer than that, so a holder stuck on one await never keeps the
  other replicas out (cobalt prod, 2026-10-31: a catch-up's first read beside a write never answered
  and its claim, renewed by its own task, held every catch-up off all night).
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
- **A stage moves the pointer at the epoch the writer saw.** A stage is the change from the scope's
  rows at `carry_at` (the epoch the writer read before it began, which its commit replaced) to its
  rows at the write's epoch. A table it did not change keeps the chunk of the pointer at `carry_at`;
  any other pointer may hold a dead write's rows that L3 has and no chunk does. The move is a
  compare-and-set against exactly the entry it was judged on; one lost to a racing writer is judged
  again against the entry the racer left, never skipped for it.
- **A pointer behind is carried to it through the stages kept** (below), and a pointer at the stage's
  epoch or past it is owed nothing.
- **Anything else goes to L3.** With no kept stage reaching `carry_at` and no earlier write still
  publishing, the tables the stage left out are read from L3 at its epoch, while L3 holds the scope
  exactly there; else `catch_up_from_l3` republishes it, the slow path but the correct one. A writer
  that finds an earlier write unfinished stages every state but those the unfinished write touched
  (its `EpochSnapshot.pending`), which go to the catch-up: their L3 rows hold the dead write's rows,
  which no chunk does.
- **The pending set is read before the write begins.** `EpochSnapshot.pending` is a set of scopes
  without the version that touched each, which a writer reads before it begins (the reports writer at
  its `prepare`). A writer whose lease lapsed but is still writing may `touch` a scope after that
  read and before the new writer's `begin` (its fence holds until `begin` takes the version), so
  such a scope is staged as if untouched, and a table its stage left out keeps a chunk without the
  dead write's rows. After `begin` no earlier write can touch
  anything (its fence then refuses it, and `begin` is refused while one of its data transactions is
  open), so the set to trust is the one read under `begin`'s lock, or re-read once it commits;
  `ScopeEpochs` does not answer that set today.
- **Hold the claim across commit and publish.** Otherwise a replica waiting on the write rebuilds
  every scope from L3 in the moment between them.

### Overlapping writers: a stage is kept, never thrown away while it can still be published

**Decided 2026-11-01, for the reports pod's AP and LSF loads.** On cobalt prod on 2026-10-31, 59
staged publishes left 2,034 scopes (about 34 a write: every scope each touched) for the catch-up,
which re-read each whole from L3 through 1,000-row pages at about 1.8 s a page (IA's 434,000 rows,
about 13 minutes), though the writer had its rows seconds before. Every one of them was a scope whose
pointer was behind the epoch its writer saw: a write left unfinished at 17:41 UTC (a deploy) staged
nothing, every later partial write of the same scopes found the pointer behind and threw its stage
away, and no catch-up brought them back while the writes ran back to back. 0.87 read such a stage's left-out tables from L3 instead (46 times
in the hour after its deploy, once per scope, inside the write's publish). A stage that cannot move
its pointer yet is now kept for the publish that can:

- **Kept, not deleted.** A committed write's stage whose pointer is behind `carry_at` is written to
  the pointer bucket as `{name}.r.{token}.{epoch}`: its chunks, its rows, which tables are pieces,
  its columns, and `carry_at` as its base. The sweep spares the chunks a kept stage names, as it
  spares the chunks a pointer serves. Only a committed write keeps a stage (it is kept by
  `publish_staged`, after the commit), so nothing kept is a write that never commits.
- **Carried on in epoch order.** A kept stage moves a pointer at its base, and only that. A publish
  that finds its pointer behind carries it to its own base through the kept stages first: each step
  takes the kept stage of the lowest epoch whose base is the pointer's epoch (two stages on one base
  cannot both be right, and only the lower can be). Every publish, staged or whole, carries its
  scopes on through the stages kept after it once it has moved them, and the catch-up does the same
  before it reads a scope from L3, reading nothing for a scope its kept stages bring to L3's epoch.
  So pieces chain in the order their writes committed, every pointer's chain stays L3's rows at its
  epoch, and a pointer never moves back.
- **Kept before the pointer is read again.** A writer keeps its stage and only then reads the pointer
  once more; a publish lists the kept stages only after its own move. So either the writer sees the
  move and carries itself, or the publish lists the stage and carries it: neither can miss the other.
- **Waits only on a write still publishing.** A kept stage whose chain back from `carry_at` (through
  the kept stages, each base the epoch of the write before) reaches an epoch with no stage kept, a
  write whose claim is still live, waits for it: in neither list `publish_staged` answers. Else its
  tables are read from L3 as before. A kept stage nothing brings to its base stays kept until a
  pointer reaches its epoch (the catch-up's L3 read), then goes: it never moves a pointer it does not
  follow.
- **Two code versions.** A replica that predates kept stages ignores their keys (its watch wakes on
  pointers and the index only), carries nothing on, and its sweep may delete a kept stage's chunks.
  A pointer is never moved onto a missing chunk, so a stage that lost one is read from L3, as before.

Measured with that pattern in memory (an AP write whose publish lags, N LSF writes committed behind it,
each published while the next is writing; 50 scopes of 3,000 whole rows and 2,000 appended rows; an L3
that pages 1,000 rows a statement; macOS arm64, 2026-11-01):

| | Scopes left for the catch-up | L3 pages (rows) read | Pointers at L3's epoch | A replica's L1 at L3 |
|---|---|---|---|---|
| 0.88, N = 1 / 3 / 10, 18 ms a page | 50 / 150 / 500 | 350 (250,300 to 250,750) | 4.1 / 3.7 / 7.5 s | 4.1 / 3.7 / 7.5 s |
| now, N = 1 / 3 / 10, 18 ms a page | 0 | 0 | 0.1 / 0.1 / 1.4 s | 2.2 / 1.8 / 3.7 s |
| 0.88, N = 3, 180 ms a page | 150 | 350 (250,400) | 9.4 s | 9.4 s |
| now, N = 3, 180 ms a page | 0 | 0 | 0.5 s | 3.9 s |

The old cost grows with L3's page time (about 66 s at prod's 1.8 s a page, read 7 scopes at once) and with
a scope's rows; the new one does not depend on L3 at all. What remains of a replica's time is its own
DuckDB swap of 50 scopes, which the old path paid too, spread across its slower publish.

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
  holds the change. A stage whose pointer is behind is kept and carried on by the publish that brings
  the pointer to its base (above); one nothing can carry leaves the scope with its pointer behind L3's
  epoch, which `catch_up_from_l3` republishes. A new scope's first piece is its base.
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
  at the same epoch (0.72's own rule; from 0.78.1 a rebuild does repoint it, see "Two code versions
  at once" below). It never decodes a piece as
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
  so it is held as the L1 keeps it: typed in Arrow, and cut by a backend's retention. The next page is
  fetched while this one is typed. Pages are never typed several at once: a retention judges the rows
  it is given, so a different cut keeps different rows. The scope is then published like any
  whole publish (this L1, its chunks, its pointer) and let go. At most `l3_concurrency` scopes
  (default 4) are held at once, read beside the one being published, and published in scope order.
- **A page reads a page, not the scope (2026-11-01).** Each page is `ORDER BY` its paging columns
  within the scope. On YugabyteDB a key led by a hashed column the scope does not pin has no index
  serving that order, so each page read and sorted the scope's every row, and a scope's read grew with
  the square of its rows (cobalt prod: IA, 434k rows, 1.8 s a page). A table whose L3 index led by the
  scope column orders its rows by other columns names them (`SnapshotTable.read_order`); its pages
  are read in that order, the rest of the key after it, and the scope is sorted back into key order
  once read.
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
| Named by a kept stage of its scope | Kept: a publish will carry it onto the pointer |
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
- **A rebuild repoints a same-epoch pointer of other columns to its own.** Otherwise a scope no
  write touched after a column change would keep the old version's pointer for good, and every
  replica would rebuild every such scope from L3 at every start. A ready replica never takes a
  same-epoch pointer of other columns (it is not ahead of what it holds), so the version still
  running keeps serving its own L1 and is not made to rebuild; its status says how many scopes'
  pointers carry other columns. Only a replica that starts while both run repoints, so the
  repointing ends with the deploy.
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
