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
| Install | `3tears[snapshot]` (pyarrow and duckdb). The store, its slots and hand-offs, and the pool need pyarrow alone; `MappedTables.connect`, `SharedArrowStore.reader` and `publish_query` also need duckdb. |
| Import | `from threetears.core.shared_arrow import SharedArrowStore, SharedArrowPool, JobContext, Parts, HandedTables, merge_parts, merged_rows` |
| Where the files live | One directory on a memory-backed filesystem: `/dev/shm`, or a Kubernetes `emptyDir` with `medium: Memory`. Never a disk. |
| Owner | One `SharedArrowStore` per directory, in the serving process. Workers never create or delete anything but their own result slot. |
| File format | Arrow IPC file format, uncompressed, one file per table, or per part for a table given as `Parts`. A mapped column is the file's bytes. |
| Layout | `versions/<name>/` holds a version's `MANIFEST` (the files it names) and its lock file; the files live in `parts/<one directory per publish>/`, never written again. Read a version's files with `version_files(root, version)`, never by listing its directory. |
| Bound | `max_bytes` on the store covers every version, result and hand-off together. Set it below the volume's `sizeLimit`: a pod whose `emptyDir` passes its `sizeLimit` is evicted. |
| What is charged to the pod | The files' pages, once (tmpfs pages count against the pod's memory limit), plus each process's private memory. |
| Cleanup | By state, never by time: a version goes when it is not current and nobody holds it, and a part file when the last version naming it goes; a result goes when it is closed and no table taken from it lives, or its job ends without a claim. A new store clears what no live process holds, part files no kept version names among it. |
| Blocking calls | `publish`, `publish_query`, `claim`, `sweep`, `settle`, `reserve`, `discard`, `hand`, `release`, `handing`, a lease's `release`, a segment's `close`, and a reader's `read` write, scan, map or remove files (most also sweep the directory): call them from a thread (`asyncio.to_thread`), never on the event loop. `SharedArrowPool.run` does this for you, and lets go of what a caller that stops waiting was handed. |
| Shared files | A version published from a `base` names the base's files it does not rewrite: they exist once, the store counts them once, and the publish touches none of them. |
| What a publish costs | The files it writes and one manifest, whatever the version holds besides. The sweep it leaves (readers told, superseded versions retired, their files removed) runs after it returns, on the store's own thread, one at a time; `store.settle()` waits for it. A publish short of room sweeps first itself. |
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

# named by the epochs of the data it holds (scope -> epoch), so the same data is the same version
async def on_change(epochs: dict[str, int]) -> None:
    def publish() -> None:
        store.publish_query(epochs_version(epochs), l1.cursor(), {"races": "SELECT * FROM races"})
    await asyncio.to_thread(publish)
```

- A table is a `pyarrow.Table`, a `pyarrow.RecordBatchReader`, or a no-argument callable returning either, opened only
  when the write reaches it (a DuckDB cursor holds one answer at a time).
- The newest version published is current. Publishing a name already present makes it current without rewriting it.
- A version is staged under a hidden name and renamed into place whole, so a reader never sees half of one.
- `SharedArrowFullError` means the store would pass `max_bytes`; nothing of the refused version is kept and the
  current one keeps serving.
- A copy that is only written and never read by the serving process can be held compressed:
  `DuckDBBackend(compressed=True)` keeps its tables in a DuckDB catalog attached with `COMPRESS` and checkpoints after
  each write on a thread of its own, off the write's path (the reports' real tables: a quarter of the memory).

## Publish a change a part at a time

A table given as `Parts` is one file per part (a state's rows, say). Publish the next version from the current one
with `base=`, naming only the parts that changed: every other part, and every table not named, is the base's own file,
named in the new version's manifest, so a change costs one part's write and one part's memory, at 60 parts or 6,000. Its first read maps only those parts too: the
serving process's reader and a worker's kept version take every file the two versions share from the mapping they
had.

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
- A version's directory holds its manifest, not its files. To read a part's file yourself (to link it again, to size
  it), hold the version and look it up:

```python
from threetears.core.shared_arrow import version_files

with store.lease("v2"):
    files = version_files(store.root, "v2")      # {"races@GA.arrow": VersionFile(path, nbytes), ...}
    ga = files["races@GA.arrow"].path            # map or open it read-only; never write, rename or remove it
    held = sum(each.nbytes for each in files.values())
```

## Merge a group of parts into one

A table kept as many small parts (one per load run, say) merges neighbouring parts into one, so readers map fewer
files and a publish names fewer parts. `merge_parts` returns the `Parts` to publish on the version: the group's parts
are dropped, every other part is the base's own file, and the merged part is written from the group's
mapped files as the publish reaches it. A merge writes the group's bytes once and nothing else.

```python
from threetears.core.shared_arrow import merge_parts

# runs r0000 .. r0009 of v7 become one part, r0000.1; the rest of v7 is kept as it is
store.publish("v8", {"lsf": merge_parts(store, "v7", "lsf", [f"r{i:04d}" for i in range(10)], "r0000.1")}, base="v7")

# a level up: merged parts merge again the same way
store.publish("v9", {"lsf": merge_parts(store, "v8", "lsf", ["r0000.1", "r0010.1"], "r0000.2")}, base="v8")
```

- Readers see the same rows in the same order before and after. That holds because a merge takes neighbours only:
  the group must be a run in part-name order, and the merged name must sort into the run's place (`f"{first}.1"`
  sorts after `first` and before the next part, since `.` sorts before every digit and letter). Anything else is
  refused (`ValueError`), as is a group of fewer than two parts; a part the version does not hold raises `LookupError`.
- Run it where the store is owned (the serving process, on a thread: the write is Arrow IPC and releases the
  interpreter lock).
- In a worker process, merge into a result instead. `merged_rows(root, version, table, names)` opens the group's
  rows as one reader and holds the version's shared lock while it reads (the serving process holds a lease on the
  version for the job, as for any read). Write it into the job's slot, then link the claimed file as the part:

```python
# worker process
slot.write({"merged": merged_rows(root, "v7", "lsf", names)()})

# serving process, on a thread
segment = store.claim(slot)
store.publish("v8", {"lsf": Parts(schema, {**dict.fromkeys(names), "r0000.1": segment.files["merged"]})}, base="v7")
segment.close()  # the version holds its own link to the file
```

- The tiering policy (which groups merge, how many levels, rewriting only the merged part that holds a dropped run)
  is the caller's. These two calls are the only primitive it needs.

## Know which columns a write moved

A memo over a version (reuse a query's answer while what it read did not move) needs to know which columns of a part
a write changed. Stamp each part as the worker writes it: a column keeps the token of the base version's same part
only when that part holds as many rows and the column is equal to it position by position (values, nulls, type);
every other column, and every column of a part with no base, takes a fresh token. A reader then prints a read from
the files' footers alone, no row read again.

```python
from threetears.core.shared_arrow import column_print, stamping, version_files

# worker process (the job holds a lease on v1): a step's name "races.GA" is v1's file races@GA.arrow
base = version_files(root, "v1")

def base_of(name: str):
    held = base.get(name.replace(".", "@", 1) + ".arrow")
    return held.path if held else None

with slot.writing() as write:
    stamping(write, base_of)({"races.GA": ga_rows})    # a pa.Table is stamped; a reader or an opener is not

# serving process, on a thread: link the claimed file as the part, then print what a query reads of it
segment = store.claim(slot)
store.publish("v2", {"races": Parts(schema, {"GA": segment.files["races.GA"]})}, base="v1")
segment.close()
with store.lease("v2"):
    printed = column_print(version_files(store.root, "v2")["races@GA.arrow"].path, ["state", "votes"])
```

- A print is `"<row token>|column=token;..."`. A file holding no tokens (a cold start's publish, a merge, a file
  written before tokens) prints as its identity: any write moves it.
- Two reads whose parts' row tokens and tokens of the columns read are equal read the same rows of those columns, in
  the same order.
- `file_tokens(path)` reads a file's footer once per process (kept by path, inode, size and modified time). It
  blocks: call it off the loop. Stamping compares in Arrow's C++ over the mapped base: run it in the worker.

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
- A worker may write its result in steps (`with slot.writing() as write: write({...}); write({...})`), so it holds one
  step's rows at a time; the slot's `max_bytes` bounds the steps together, and the result is the slot's only as the
  block ends. A step naming a table an earlier one wrote is refused.
- Close the segment when done. Tables taken from it are its files' pages (zero-copy), valid after `close` too: one
  kept after `close` (or a slice of it, or anything sharing its buffers) keeps its files in the store, counted
  against `max_bytes`, until the last buffer is freed; the first sweep after that removes them. So `max_bytes` bounds
  what results hold in memory however long a caller keeps them, and a result held too long refuses the next publish
  (`SharedArrowFullError`) rather than pass the bound.
- `private_bytes()` (same module) is the private memory a worker is judged by: anonymous resident pages, never the
  shared files it maps. Judge any worker of your own over the store by it, not by resident memory.

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

`status()` reads the store's own tallies, never the directory and never its lock, so call it from the event loop.
Its `held_bytes`, `versions` and `handoffs_held` are what `store.held_bytes()`, `store.versions()` and
`store.handoffs()` find by walking the directory (a file linked into several versions counted once), except for
what is still being written: a version or hand-off counts once it is renamed into place, and a worker's result
once it is claimed (until then its open slot shows in `slots_open`). The three walking methods block: call them on a
thread. Right after a publish, the sweep it left may still be running on the store's thread: call `store.settle()`
(on a thread) first to read the directory as that sweep leaves it.

## What it saves, measured

Measured 2026-10-30 by a script retired with the scoped snapshot (`scripts/measure-shared-arrow.py` in the git
history). It loaded a spec'd set of tables into DuckDB, published them, and had N workers scan
every column at once, reporting each process's PSS (`/proc/<pid>/smaps_rollup`): one process holding the copy
(today), that process beside the store and workers (`shared`), each worker loading its own copy (`naive`), and the
serving process reading the store itself with its DuckDB copy held compressed only to take changes (`store`), on Linux
with the store on `/dev/shm`.

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
