---
artifact: build-plan
version: 1
scope: shared-arrow
branch: feature/shared-memory-snapshots
depends_on: []
governed_by: []
last_validated: 2026-10-30
---

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

### Chunk 01: Store, versions, leases, sweep

Pool with leases; tests first (multi-process readers of one mapping with a Linux PSS check, swap with a reader
mid-read, retirement only after every reader lets go, bounded). Done.

### Chunk 02: Results direction

Slots, claim/discard, crash and abandon paths; guide `docs/shared-arrow.md`. Done.

### Chunk 03: Measurement

`scripts/measure-shared-arrow.py` and real-size figures. Done.

### Chunk 04: Reports adoption

A separate PR in the reports repo, after a 3tears release carrying this (reports
`.prawduct/artifacts/build-plan-compute-workers.md`). Not in this PR.

### Chunk 05: The serving process reads the store and drops its own copy

Pace, 2026-10-30: total memory about 1x today, not 1.8x. `SharedArrowReader` (`store.reader()`), `Parts` +
`publish(base=)` by hard link, `SharedSnapshot` (a scoped snapshot published a scope's part at a time),
`ScopedSnapshot.behind`, `DuckDBBackend(compressed=True)` with its checkpoint on a thread of its own. Done.

### Chunk 06: Critic findings (rev-20261030T131014Z-138eb296)

A caller cancelled during the pool's submit or claim hop lets go of the slot or the claimed result; a slot reserved
during a publish takes its room from it; the sweep removes files outside the store's lock; every writer checks part
names; one naming rule (`plain_name`) and one query helper (`query_batches`) shared with `SharedSnapshot`; refused
slots and dead workers logged; the checkpointer sees a stop set at any moment and holds its backend weakly. Done.

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

Chunk 05 decisions:

- [DECISION] The DuckDB copy is not deleted: it is what applies each change (scope swaps, the reports' retention,
  derived columns, rows kept apart), and Arrow files are immutable. It is held compressed (`ATTACH ':memory:'
  (COMPRESS)`: real election extracts 513 -> 116 MiB) and never read by requests. Rewriting the snapshot backend on
  Arrow was the alternative; too large for the gain. Told to Pace's session before building.
- [DECISION] A version reuses the last one's unchanged parts by hard link (same inode: one set of pages, counted once
  by inode against `max_bytes`), so a change costs one scope's write, not the copy's.
- [DECISION] The checkpoint that compresses a change runs on the backend's own thread after the write lets go of the
  lock: off the change's path (coordinator: the <40 s update budget must not get worse).
- [DECISION] `SharedSnapshot` writes a scope again when its epoch moved, when no epoch names it, or when the base
  lacks it; reads each table's scope values every publish, so nothing the copy holds is left out.
- [DECISION] Publish and sweep race closed: the version being published is named to the sweep before its rename
  (`_pending`), so a lease released on another thread cannot retire it before it is current.

Chunk 06 evidence. Mutations (scratch, restored), each failing its own new test: submit hop not shielded; claim hop
not shielded; publish room fixed at its start; refused slot not logged; rmtree under the lock; no part check in the
writer; worker death silent; the old scope naming rule; stop read before the clear; checkpointer holding the backend
strongly.

Chunk 05 evidence. Mutations (scratch, restored): hard links made copies -> 3 tests fail; inode dedupe removed -> 2;
reader keeps a superseded version -> 4; no `_pending` guard -> 1; readers not told at publish -> 1; every scope
rewritten -> 1; no background checkpoint -> 3; catalog attached without COMPRESS -> 1.

Measured, 2026-10-30 (Linux arm64 container, `/dev/shm`, each state's rows side by side):

| layout | 2 workers | 4 workers |
|---|---|---|
| today: one process holding the copy (PSS) | 2.17 GiB | 2.18 GiB |
| shared: serving process + store + workers | 4.01 GiB | 4.18 GiB |
| store: serving reads the store, copy compressed (0.18 GiB) + workers | 2.30 GiB | 2.57 GiB |

Generated values compress to a twelfth, real ones to a quarter: a real copy adds about 0.3 GiB. One-state change:
write plain 0.08-0.10 s vs compressed 0.08-0.10 s (LSF), 0.04-0.05 s both (AP); checkpoint 0.17-0.21 / 0.01-0.02 s,
off the path; publish of the state 0.10-0.15 s, on the path. Every state published: 2.3-5.6 s (a replica's start).

## Where this stands (2026-10-30, paused for the prod release)

- Head `d05e955f2` on `feature/shared-memory-snapshots`. Chunks 01-06 built.
- Critic `rev-20261030T131014Z-138eb296`: 0 blocking. R-2 to R-13 fixed, and verify-resolutions
  `rev-20261030T133120Z-1cc9aecb` marked them fixed with 0 findings. R-14 to R-16 were informational and are
  accepted. O-1 to O-3 are accepted on the record.
- R-1 is still open: test evidence is not recorded.
- The full CI run after the verify (CI script, as the workflow runs it) failed `test_public_api_has_a_caller` and
  `test_slf001_pragma_policy`. Commit `d05e955f2` fixes both: tests reach the code through its front door
  (`checkpoint_when_asked`, `DuckDBBackend.checkpoint`), `scope_part` is now private, and the ledger names the reports
  pod as the caller of `SharedArrowReader`, `SharedRead` and `SharedSnapshot`. No Critic pass has run on `d05e955f2`.
- CI on `d05e955f2` passed lint, typecheck, unit + enforcement, otel floor and mcp 1.x. Integration ran 498 passed and
  1 failed: `packages/nats/tests/integration/test_credential_renewal_live.py::test_many_renewals_under_a_busy_subscription_lose_and_double_nothing`.
  Integration passed in the previous run on `623dddfdb` (499 passed), and this branch changes nothing in nats.
  Re-run it before calling the branch green.
- Next steps:
  1. Re-run integration, or the whole CI script, on the head commit and confirm green.
  2. Record test evidence: `prawduct-hook test-evidence record`, which closes R-1.
  3. Update the PR #1050 body from `pr1050-body.md` in the scratchpad and fill in the CI line.
  4. Tell the coordinator when it is green; Pace merges.
  5. 0.86.0 is held until Pace says go. Two more 3tears changes join that release: tool-call metrics, and the
     registry not being ready until one heartbeat round.
- Reports branch `feature/report-computes-in-workers` (worktree `../reports-wt-shm-workers`): the serving move is
  built and committed, and develop is merged in (#177 cap: a worker fallback goes through `computed`). It waits on
  0.86.0 for its re-pin. See its build plan, chunk 02.
