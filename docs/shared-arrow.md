# Shared-memory Arrow: worker processes without copies

`threetears.core.shared_arrow` lets a serving process hand data to worker processes, and take their results back,
through Arrow files in a memory-backed directory. Every process maps the same pages, so N workers cost one copy of
the data, and a worker's result tables come back with no pickle and no copy. The serving process can read the same
pages itself, so the data costs one copy in all.

Use it when CPU work must leave the serving process (a thread holds the interpreter lock and stalls the event loop)
and that work reads, or produces, more data than you want copied per process.

## Facts every section relies on

| Fact | Value |
|---|---|
| Install | `3tears[snapshot]` (pyarrow and duckdb) |
| Import | `from threetears.core.shared_arrow import SharedArrowStore, SharedArrowPool, JobContext, Parts, HandedTables` |
| Where the files live | One directory on a memory-backed filesystem: `/dev/shm`, or a Kubernetes `emptyDir` with `medium: Memory`. Never a disk. |
| Owner | One `SharedArrowStore` per directory, in the serving process. Workers never create or delete anything but their own result slot. |
| File format | Arrow IPC file format, uncompressed, one file per table, or per part for a table given as `Parts`. A mapped column is the file's bytes. |
| Bound | `max_bytes` on the store covers every version, result and hand-off together. Set it below the volume's `sizeLimit`: a pod whose `emptyDir` passes its `sizeLimit` is evicted. |
| What is charged to the pod | The files' pages, once (tmpfs pages count against the pod's memory limit), plus each process's private memory. |
| Cleanup | By state, never by time: a version goes when it is not current and nobody holds it; a result goes when it is closed or its job ends without a claim. A new store clears what no live process holds. |
| Blocking calls | `publish`, `publish_query`, `claim`, `sweep`, `reserve`, `discard`, `hand`, `release`, `handing`, a lease's `release`, a segment's `close`, a reader's `read` and `SharedSnapshot.publish` write, scan, map or remove files (each of the last five sweeps the directory): call them from a thread (`asyncio.to_thread`), never on the event loop. `SharedArrowPool.run` does this for you, and lets go of what a caller that stops waiting was handed. |
| Shared files | A version published from a `base` hard-links the files it does not rewrite: they exist once, and the store counts them once. |
| Platform | Linux and macOS (`flock`, `mmap`). PSS measurements need Linux. |

## Give the volume to the pod

```yaml
spec:
  containers:
    - name: app
      env:
        - name: SHARED_ARROW_ROOT
          value: /shared-arrow
      volumeMounts:
        - name: shared-arrow
          mountPath: /shared-arrow
  volumes:
    - name: shared-arrow
      emptyDir:
        medium: Memory
        sizeLimit: 3Gi      # the store's max_bytes sits below this
```

The volume's pages count against the container's memory limit, so raise the limit by what the store holds (one
version, plus results in flight), not by one copy per worker.

## Publish a data version for workers to read

```python
import asyncio
import duckdb
import pyarrow as pa
from threetears.core.shared_arrow import SharedArrowStore, epochs_version

store = SharedArrowStore("/dev/shm/my-app", max_bytes=2 << 30)

# from Arrow tables you already hold
store.publish("v1", {"races": pa.table({"state": ["VA", "GA"], "votes": [10, 20]})})

# from a DuckDB copy, a batch at a time (no table is held whole in this process)
l1 = duckdb.connect()
l1.execute("CREATE TABLE races AS SELECT 'VA' AS state, 10 AS votes")
store.publish_query("v2", l1.cursor(), {"races": "SELECT * FROM races"})

# named by a scoped snapshot's epochs, so the same data is the same version
async def on_change(snapshot) -> None:
    def publish() -> None:
        with snapshot.read_versioned() as read:
            store.publish_query(epochs_version(read.epochs), read.cursor, {"races": "SELECT * FROM races"})
    await asyncio.to_thread(publish)
```

- A table is a `pyarrow.Table`, a `pyarrow.RecordBatchReader`, or a no-argument callable returning either, opened only
  when the write reaches it (a DuckDB cursor holds one answer at a time).
- The newest version published is current. Publishing a name already present makes it current without rewriting it.
- A version is staged under a hidden name and renamed into place whole, so a reader never sees half of one.
- `SharedArrowFullError` means the store would pass `max_bytes`; nothing of the refused version is kept and the
  current one keeps serving.

## Publish a change a part at a time

A table given as `Parts` is one file per part (a state's rows, say). Publish the next version from the current one
with `base=`, naming only the parts that changed: every other part, and every table not named, is the base's own file,
hard-linked, so a change costs one part's write and one part's memory.

```python
import pyarrow as pa
from threetears.core.shared_arrow import Parts, SharedArrowStore

store = SharedArrowStore("/dev/shm/my-app", max_bytes=2 << 30)
schema = pa.schema([("state", pa.string()), ("votes", pa.int64())])

store.publish("v1", {"races": Parts(schema, {
    "VA": pa.table({"state": ["VA"], "votes": [10]}, schema=schema),
    "GA": pa.table({"state": ["GA"], "votes": [20]}, schema=schema),
})})

# GA moved and TX is gone; VA is v1's file
store.publish("v2", {"races": Parts(schema, {
    "GA": pa.table({"state": ["GA"], "votes": [25]}, schema=schema),
    "TX": None,
})}, base="v1")
```

- Readers see one table: its parts' rows in part-name order, no row copied.
- Every part must have exactly the table's `schema`; a table whose columns changed from the base's is refused
  (`ValueError`): publish it without `base`.
- `base` is held for the write; a base the store no longer holds raises `LookupError`.

## Publish a scoped snapshot's copy

`threetears.core.collections.shared_snapshot.SharedSnapshot` publishes a scoped snapshot's L1 a scope's part at a
time: each version is written from one read of the copy, and only the scopes whose epoch moved since the version it
published last are written again.

```python
import asyncio
from threetears.core.collections.shared_snapshot import SharedSnapshot

shared = SharedSnapshot(store, snapshot_tables)          # the snapshot's SnapshotTable declarations


async def on_change() -> None:
    def publish() -> None:
        with snapshot.read_versioned() as read:
            shared.publish(read)                          # named by the read's epochs unless version_name= says
    await asyncio.to_thread(publish)
```

- `extra_tables=` adds tables written whole into every version, from the read (what its epochs are, say).
- A store whose current version this publisher did not make is written whole.
- A copy that is only written and never read by the serving process can be held compressed:
  `DuckDBBackend(compressed=True)` keeps its tables in a DuckDB catalog attached with `COMPRESS` and checkpoints after
  each write on a thread of its own, off the write's path (the reports' real tables: a quarter of the memory).

## Read the current version in the serving process

The process that owns the store reads the current version in place through a reader: the same pages its workers map,
no copy of its own.

```python
import asyncio

reader = store.reader(config={"threads": 2})


def total(state: str) -> int:
    with reader.read() as read:                           # the version current as the read opens
        return read.cursor.execute("SELECT sum(votes) FROM races WHERE state = ?", [state]).fetchone()[0]


async def handler(state: str) -> int:
    return await asyncio.to_thread(total, state)
```

- A read keeps its version for its block, whatever is published meanwhile; the next read takes the new one.
- The store tells its readers after each publish: a superseded version goes as soon as its last read ends, so two
  versions are held only while a read of the old one is open.
- `read.cursor` is a DuckDB cursor with every table as a view over its mapped Arrow; `read.tables` the tables.
- `reader.status()` gives the version held, its open reads, and the reads and mappings so far. Close the reader when
  done.

## Run jobs in workers over the current version

```python
import asyncio
import pyarrow as pa
from threetears.core.shared_arrow import JobContext, SharedArrowPool, SharedArrowStore


def total_votes(context: JobContext, state: str) -> int:
    """runs in a worker; a module function, so the worker imports it by name."""
    connection = context.version.connect(config={"threads": 2})
    try:
        return connection.execute("SELECT sum(votes) FROM races WHERE state = ?", [state]).fetchone()[0]
    finally:
        connection.close()


async def main() -> None:
    store = SharedArrowStore("/dev/shm/my-app", max_bytes=2 << 30)
    await asyncio.to_thread(store.publish, "v1", {"races": pa.table({"state": ["VA"], "votes": [10]})})
    pool = SharedArrowPool(store, workers=2, private_bytes=1 << 30, name="reports")
    try:
        result = await pool.run(total_votes, "VA")
        print(result.value)
    finally:
        await pool.close()
```

- Workers are spawned (never forked) once and kept. One whose private memory passes `private_bytes` after a job is
  replaced, so what it allocated goes back to the system. Pass `initializer=` to size its thread pools to its CPU
  share.
- `run` leases the version current at submission (or `read="v1"` for a named one, `read=False` for none) and holds it
  until the worker is done with it, even if the caller stops waiting. A newer version published meanwhile does not
  disturb the job.
- `context.version.tables` gives the mapped `pyarrow.Table`s; `context.version.connect(schema=...)` gives a DuckDB
  connection with each table as a view, no row copied.
- A worker keeps its last version mapped between jobs and lets it go after a job that finds a newer one current.
- What the job raises is raised by `run`. A worker that dies raises `BrokenProcessPool` and its pool is replaced.

## Hand a worker's result tables back without a pickle

```python
import asyncio

import pyarrow as pa
from threetears.core.shared_arrow import JobContext


def build_rows(context: JobContext, state: str) -> int:
    rows = pa.table({"state": [state] * 1_000_000, "n": range(1_000_000)})
    context.result.write({"rows": rows})   # into the slot the pool reserved
    return rows.num_rows                   # the small value still comes back pickled


async def caller(pool) -> None:
    result = await pool.run(build_rows, "VA", read=False, result_bytes=256 << 20)
    segment = result.segment                 # mapped here
    try:
        print(segment.tables["rows"].num_rows, result.value)
    finally:
        await asyncio.to_thread(segment.close)   # closing removes its files: a thread, not the loop
```

- `result_bytes` reserves a slot counted at that size against `max_bytes` until it is claimed or discarded; a store
  without room raises `SharedArrowFullError` before the job runs.
- A worker writing more than its slot holds gets `SharedArrowFullError` in the worker, raised by `run`; nothing is kept.
- The serving process owns the slot from reservation to release. A job that raises, a caller that stops waiting (at
  any point: while the job is submitted, run, or its result claimed) and a worker that dies mid-write all end with
  the slot discarded, or the claimed result closed, once the worker is done; a dead writer's staging directory is
  removed because its `flock` died with it. A worker's death is logged once per pool it breaks.
- A slot reserved while a version is being published takes its room from that publish, never from the bound: the
  publish is refused (`SharedArrowFullError`) rather than pass `max_bytes`. A refused slot is logged.
- A job given a slot may write nothing: `result.segment` is then `None`.
- Close the segment when done. Tables taken from it stay valid while it is open; one kept after `close` keeps its
  pages mapped until it is dropped, but its files are already gone from the store's count.

## Hand a worker its input tables without a pickle

A `ProcessPoolExecutor` pickles a job's arguments whole, in one call on its feeder thread, which never lets the
interpreter lock go: a 68 MiB Arrow table held the serving process's event loop 74-165 ms on cobalt prod. Hand the
tables through the store instead: the serving process writes them as Arrow IPC (pyarrow releases the lock for each
batch), and only their name crosses.

```python
import pyarrow as pa
from threetears.core.shared_arrow import JobContext


def apply_change(context: JobContext, state: str) -> int:
    rows = context.tables.tables["rows"]     # mapped read-only, no row copied
    return rows.num_rows


async def caller(pool, rows: pa.Table) -> None:
    result = await pool.run(apply_change, "VA", read=False, tables={"rows": rows})
    print(result.value)
```

- The hand-off is written on the pool's thread, never the event loop, and counts against `max_bytes` (at its bound
  while written, then at its files' bytes). A store without room raises `SharedArrowFullError` before the job runs.
- Its bound is its tables' bytes plus the file's own; name `tables_bytes` (or `max_bytes` on `hand`) to set it, and
  always for a `RecordBatchReader`, whose size is not known before it is written.
- It is released when the job's future is done: the job returned, raised, or its worker died. Its files go at once,
  or, when a worker still maps it, at the first sweep after the worker lets go (its `flock` dies with it).
- `context.tables` is valid for the job's run only: copy what the job keeps past it.

With your own executor, wait inside `handing` for the job, and map in the worker:

```python
from threetears.core.shared_arrow import HandedTables


# serving process, on a thread
with store.handing({"rows": rows}) as handed:   # released as the block ends, however the job ended
    future = my_executor.submit(my_job, handed, "VA")
    count = future.result()


# worker process
def my_job(handed: HandedTables, state: str) -> int:
    with handed.mapped() as mapped:            # holds the hand-off until closed
        return mapped.tables["rows"].num_rows
```

A hand-off released before its worker maps it raises `LookupError` in the worker.

## Use the store with your own process pool

The pool is optional. Any channel that carries a pickled argument works:

```python
from threetears.core.shared_arrow import read_version

# serving process
lease = store.lease()                      # hold the current version for the job
slot = store.reserve(64 << 20)             # a place for its result
future = my_executor.submit(my_job, str(store.root), lease.version.name, slot)
try:
    future.result()
    segment = store.claim(slot)            # None if the job wrote nothing
finally:
    lease.release()
    store.discard(slot)                    # harmless after a claim


# worker process
def my_job(root: str, version: str, slot) -> None:
    with read_version(root, version) as mapped:
        slot.write({"out": mapped.tables["races"].slice(0, 10)})
```

Hold the lease until the worker is done reading: it is what keeps the version from being retired between the hand-off
and the mapping.

## Watch it

`store.status()` returns plain JSON values for a status tool: the current version, the versions held, open slots,
claimed results, open hand-offs and how many were made, `held_bytes`, `max_bytes`, and how many versions were published, retired and refused. The store logs
each publish, retirement and refusal.

## What it saves, measured

`scripts/measure-shared-arrow.py` loads a spec'd set of tables into DuckDB, publishes them, and has N workers scan
every column at once, reporting each process's PSS (`/proc/<pid>/smaps_rollup`): one process holding the copy
(today), that process beside the store and workers (`shared`), each worker loading its own copy (`naive`), and the
serving process reading the store itself with its DuckDB copy held compressed only to take changes (`store`). Run it
on Linux with the store on `/dev/shm`.

The reports pod's dev tables (9.0M rows, its real DuckDB column types, generated values, each state's rows side by
side), 2026-10-30:

| layout | 2 workers | 4 workers |
|---|---|---|
| today: one process holding the copy in DuckDB | 2.17 GiB | 2.18 GiB |
| shared: that process + the store (1.64 GiB) + workers | 4.01 GiB | 4.18 GiB |
| store: serving reads the store, copy compressed (0.18 GiB) | 2.30 GiB | 2.57 GiB |
| naive: each worker its own copy | 7.91 GiB | 11.96 GiB |

Generated values compress better in DuckDB than real ones: the reports' real extracts compress to a quarter of the
plain copy, these to a twelfth, so a real copy held compressed is nearer 0.5 GiB and the `store` layout about 0.3 GiB
more than above. A change in the `store` layout, one state replaced (the LSF table's 151k rows; or every results
table's), against a plain copy:

| change | write, plain | write, compressed | checkpoint (off the change's path) | publish of the one state |
|---|---|---|---|---|
| LSF run | 0.08-0.10 s | 0.08-0.10 s | 0.17-0.21 s | 0.11-0.15 s |
| AP load | 0.04-0.05 s | 0.04-0.05 s | 0.01-0.02 s | 0.10-0.12 s |

Publishing every state (a replica's start) took 2.3-5.6 s.
