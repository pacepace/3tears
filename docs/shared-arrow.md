# Shared-memory Arrow: worker processes without copies

`threetears.core.shared_arrow` lets a serving process hand data to worker processes, and take their results back,
through Arrow files in a memory-backed directory. Every process maps the same pages, so N workers cost one copy of
the data, and a worker's result tables come back with no pickle and no copy.

Use it when CPU work must leave the serving process (a thread holds the interpreter lock and stalls the event loop)
and that work reads, or produces, more data than you want copied per process.

## Facts every section relies on

| Fact | Value |
|---|---|
| Install | `3tears[snapshot]` (pyarrow and duckdb) |
| Import | `from threetears.core.shared_arrow import SharedArrowStore, SharedArrowPool, JobContext` |
| Where the files live | One directory on a memory-backed filesystem: `/dev/shm`, or a Kubernetes `emptyDir` with `medium: Memory`. Never a disk. |
| Owner | One `SharedArrowStore` per directory, in the serving process. Workers never create or delete anything but their own result slot. |
| File format | Arrow IPC file format, uncompressed, one file per table. A mapped column is the file's bytes. |
| Bound | `max_bytes` on the store covers every version and result together. Set it below the volume's `sizeLimit`: a pod whose `emptyDir` passes its `sizeLimit` is evicted. |
| What is charged to the pod | The files' pages, once (tmpfs pages count against the pod's memory limit), plus each process's private memory. |
| Cleanup | By state, never by time: a version goes when it is not current and nobody holds it; a result goes when it is closed or its job ends without a claim. A new store clears what no live process holds. |
| Blocking calls | `publish`, `publish_query`, `claim` and `sweep` write or scan files: call them from a thread (`asyncio.to_thread`), never on the event loop. `SharedArrowPool.run` does this for you. |
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
import pyarrow as pa
from threetears.core.shared_arrow import JobContext


def build_rows(context: JobContext, state: str) -> int:
    rows = pa.table({"state": [state] * 1_000_000, "n": range(1_000_000)})
    context.result.write({"rows": rows})   # into the slot the pool reserved
    return rows.num_rows                   # the small value still comes back pickled


async def caller(pool) -> None:
    result = await pool.run(build_rows, "VA", read=False, result_bytes=256 << 20)
    with result.segment as segment:          # mapped here; closing it frees the files
        rows = segment.tables["rows"]
        print(rows.num_rows, result.value)
```

- `result_bytes` reserves a slot counted at that size against `max_bytes` until it is claimed or discarded; a store
  without room raises `SharedArrowFullError` before the job runs.
- A worker writing more than its slot holds gets `SharedArrowFullError` in the worker, raised by `run`; nothing is kept.
- The serving process owns the slot from reservation to release. A job that raises, a caller that stops waiting and
  a worker that dies mid-write all end with the slot discarded once the worker is done; a dead writer's staging
  directory is removed because its `flock` died with it.
- A job given a slot may write nothing: `result.segment` is then `None`.
- Close the segment when done. Tables taken from it stay valid while it is open; one kept after `close` keeps its
  pages mapped until it is dropped, but its files are already gone from the store's count.

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
claimed results, `held_bytes`, `max_bytes`, and how many versions were published, retired and refused. The store logs
each publish, retirement and refusal.

## What it saves, measured

`scripts/measure-shared-arrow.py` loads a spec'd set of tables into DuckDB, publishes them, and has N workers scan
every column at once, reporting each process's PSS (`/proc/<pid>/smaps_rollup`) against one process holding the copy
and against each worker loading its own. Run it on Linux with the store on `/dev/shm`.

The reports pod's dev tables (9.0M rows, its real DuckDB column types, generated values), 2026-10-29:

| layout | 2 workers | 4 workers |
|---|---|---|
| one process holding the copy in DuckDB | 2.24 GiB | 2.25 GiB |
| that process + the store (1.75 GiB) + workers | 4.14 GiB | 4.29 GiB |
| each worker's private memory | 66 MiB | 67 MiB |
| each worker loading its own copy instead | 8.27 GiB | 12.49 GiB |

The store is one more copy beside the serving process's own, flat in the worker count; per-worker copies grow with
it. A serving process that reads the store itself instead of keeping its own copy would pay about one copy in all.
