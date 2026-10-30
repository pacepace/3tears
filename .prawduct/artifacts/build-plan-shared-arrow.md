# Build plan: shared-memory Arrow for worker processes

Feature, large (new module and API surface). Branch `feature/shared-memory-snapshots` from origin/develop @4695a8262
(0.85.0). v0.85.0 was tagged while this was built, so it ships as 0.86.0 (minor bump: new API).

## Goals

- The reports pod computes reports in worker processes (Pace: CPU work in processes, never `to_thread`; 39 loop
  stalls in 40 min, up to 261 ms, on dev) without a copy of its data in each worker: its pods sit near their memory
  alerts.
- Generic for any 3tears consumer, both directions (Pace, scope addition): (a) the serving process publishes a
  versioned read-only snapshot many workers map; (b) a worker hands Arrow results back that the serving process maps,
  no pickle, no copy, with clear ownership so nothing leaks, a worker crash included. Bounded total size; memory-
  backed only, never a disk.
- Retirement by the epoch system and reader state, never TTLs or timers.
- Measured at real size: PSS of the serving process plus N workers against today's single copy.

Decisions:

- [DECISION] One module, `threetears.core.shared_arrow`, one owner object (`SharedArrowStore`) per directory with two
  halves (`versions/`, `segments/`). The surveyed primitives (scoped snapshot's L2 chunks, DuckDB L1, complete copy)
  are per-process copies; nothing in 3tears maps memory across processes, and the reports pod's `WorkerPool`
  (product-local, listed as a 3tears candidate) moves rows over a socket. Settled by the survey.
- [DECISION] Reader liveness by `flock` (shared for readers, exclusive try for the retirer and for a slot's writer):
  the kernel drops a dead holder's lock, so a crashed worker holds nothing and nothing needs a timer. Settled by the
  no-TTL rule.
- [DECISION] Uncompressed Arrow IPC file format: a mapped buffer is the column itself; compression would make every
  reader decompress a private copy.
- [DECISION] The serving process owns every result slot from `reserve` to `claim`/`discard`; a worker only writes its
  own slot, staged under its exclusive lock and renamed whole. An open slot counts at its declared most, so the bound
  is strict rather than racy across concurrent writers.
- [DECISION] A worker keeps its last mapping between jobs (a fresh mapping costs the page faults again) and lets it go
  after any job that finds a newer version current; so an idle worker can hold one superseded version until its next
  job. Accepted bound, documented.
- [DECISION] The worker bound counts private memory (`RssAnon`), not RSS: RSS includes the shared mapped pages.
- [INFERENCE] `max_bytes` sits below the emptyDir `sizeLimit` because kubelet evicts a pod past it.

## Chunks

1. Store, versions, leases, sweep; pool with leases; tests first (multi-process readers of one mapping with a Linux
   PSS check, swap with a reader mid-read, retirement only after every reader lets go, bounded). Done.
2. Results direction (slots, claim/discard, crash and abandon paths), guide `docs/shared-arrow.md`. Done.
3. Measurement script and real-size figures. Done.
4. Reports adoption: separate PR in the reports repo, after a 3tears release carrying this.

## Evidence

Tests fail first: each guard was mutated and the suite run (scratch runs, not committed):

| mutation | tests that failed |
|---|---|
| retirement ignores readers' locks | 10 (every swap and retirement test, the bound test) |
| abandoned slots never discarded | 3 (raised, died mid-write, abandoned) |
| no bound | 3 (publish, open slot, worker write) |
| worker keeps a superseded mapping | 2 (swap tests) |
| module absent | collection error |

Measured (Linux arm64 container, store on `/dev/shm`; reports dev table shapes from `enr.load_status`, 8,982,184
rows, real DuckDB column types, generated values):

| layout | 2 workers | 4 workers |
|---|---|---|
| today: one process holding the copy (PSS) | 2.24 GiB | 2.25 GiB |
| shared: serving process + store + workers (PSS) | 4.14 GiB | 4.29 GiB |
| worker private memory each (RssAnon) | 66 MiB | 67 MiB |
| store (tmpfs) | 1.75 GiB | 1.75 GiB |
| naive: each worker its own copy (PSS) | 8.27 GiB | 12.49 GiB |

Each worker maps all 1.75 GiB (RSS 1.89 GiB) and is charged its share (PSS 0.97 GiB at 2, 0.52 GiB at 4). The store
is a second copy beside the serving process's DuckDB L1, so the pod pays about 1.8x today's data memory, flat in the
worker count; naive copies pay (1 + N)x. Publish 2.6-3.5 s, a full scan of every column 1.6-1.9 s per worker.

Follow-up worth weighing, not built: the serving process could read the store too and drop its own L1 copy for the
tables workers serve, which would bring the shared layout to about 1x.
