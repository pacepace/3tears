"""Jobs the shared Arrow tests run in spawned worker processes: module functions, so a worker imports them by name.

Each takes the job's context first, as :class:`~threetears.core.shared_arrow.SharedArrowPool` hands it.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.shared_arrow import HandedTables, JobContext, MappedTables, MappedVersion, Parts

#: how long a job waits on the test's signal file before it gives up
_WAIT_SECONDS = 60.0

#: memory a job keeps between jobs, so the worker's private memory grows
_KEPT: list[bytearray] = []


def _wait_for(path: Path) -> None:
    """wait until ``path`` exists.

    :param path: the signal file
    :ptype path: Path
    :return: nothing
    :rtype: None
    :raises TimeoutError: when it never appears
    """
    deadline = time.monotonic() + _WAIT_SECONDS
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no signal at {path}")
        time.sleep(0.01)


def _version(context: JobContext) -> MappedVersion:
    """the job's mapped version, which these jobs are always run with.

    :return: the version
    :rtype: MappedVersion
    """
    assert context.version is not None
    return context.version


def column_sum(context: JobContext, table: str, column: str) -> int:
    """the sum of one column, read from the mapped Arrow.

    :return: the sum
    :rtype: int
    """
    return int(pc.sum(_version(context).tables[table].column(column)).as_py())


def duckdb_sum(context: JobContext, table: str, column: str) -> int:
    """the sum of one column, through DuckDB over the mapped Arrow, its views in schema ``s``.

    :return: the sum
    :rtype: int
    """
    connection = _version(context).connect(schema="s", config={"threads": 1})
    try:
        found = connection.execute(f'SELECT sum("{column}") FROM s."{table}"').fetchone()
    finally:
        connection.close()
    assert found is not None
    return int(found[0])


def mapping_of(path: str) -> dict[str, int]:
    """this process's resident and proportional bytes of one mapped file (Linux ``smaps``); empty elsewhere.

    :param path: the file
    :ptype path: str
    :return: ``{"rss": ..., "pss": ...}`` in bytes, or empty
    :rtype: dict[str, int]
    """
    found = {"rss": 0, "pss": 0}
    try:
        with open("/proc/self/smaps") as smaps:
            inside = False
            for line in smaps:
                fields = line.split()
                if fields and "-" in fields[0] and len(fields) >= 5:
                    inside = fields[-1] == path
                elif inside and fields[0] in ("Rss:", "Pss:"):
                    found[fields[0][:-1].lower()] += int(fields[1]) * 1024
    except OSError:
        # NOSILENT: no /proc (macOS): the caller skips the sharing check
        return {}
    return found


def shared_read(context: JobContext, table: str, column: str, signals: str, readers: int, path: str) -> dict[str, Any]:
    """read a whole column, wait until ``readers`` jobs have each read, then report this process's share of the file.

    :return: the process id, the sum, and the file's resident and proportional bytes here
    :rtype: dict[str, Any]
    """
    total = column_sum(context, table, column)
    where = Path(signals)
    (where / f"read-{os.getpid()}").touch()
    deadline = time.monotonic() + _WAIT_SECONDS
    while len(list(where.glob("read-*"))) < readers:
        if time.monotonic() > deadline:
            raise TimeoutError("the other readers never read")
        time.sleep(0.01)
    return {"pid": os.getpid(), "sum": total, **mapping_of(os.path.realpath(path))}


def read_on_signal(context: JobContext, table: str, column: str, signals: str) -> int:
    """map the version, say so, wait for the test's go, then read: a reader mid-read across a version swap.

    :return: the sum
    :rtype: int
    """
    where = Path(signals)
    (where / "started").touch()
    _wait_for(where / "go")
    return column_sum(context, table, column)


def process_id(context: JobContext) -> int:
    """the worker's process id.

    :return: the id
    :rtype: int
    """
    return os.getpid()


def process_name(context: JobContext) -> str:
    """the worker's name as Linux says it (``/proc/self/comm``); empty elsewhere.

    :return: the name
    :rtype: str
    """
    comm = Path("/proc/self/comm")
    return comm.read_text(encoding="utf-8").strip() if comm.exists() else ""


def process_and_parent(context: JobContext) -> tuple[int, int]:
    """the worker's process id and its parent's (the fork server's, for a pool's worker).

    :return: the ids
    :rtype: tuple[int, int]
    """
    return os.getpid(), os.getppid()


def note_start(context: JobContext, started: str, name: str) -> None:
    """append ``name`` to the file ``started`` as the job starts: the order jobs were given their workers.

    :return: nothing
    :rtype: None
    """
    with open(started, "a", encoding="utf-8") as noted:
        noted.write(f"{name}\n")


def note_start_then_wait(context: JobContext, started: str, name: str, signals: str) -> None:
    """:func:`note_start`, then hold the worker until the test's go.

    :return: nothing
    :rtype: None
    """
    note_start(context, started, name)
    _wait_for(Path(signals) / "go")


def grow(context: JobContext, nbytes: int) -> int:
    """keep ``nbytes`` of private memory past the job, touched, and answer the worker's process id.

    :return: the id
    :rtype: int
    """
    kept = bytearray(nbytes)
    for index in range(0, nbytes, 4096):
        kept[index] = 1
    _KEPT.append(kept)
    return os.getpid()


def fail(context: JobContext, message: str) -> None:
    """raise.

    :raises ValueError: always
    """
    raise ValueError(message)


def make_rows(context: JobContext, rows: int, value: int) -> str:
    """write a result: a table of ``rows`` rows of ``value``, and a small one beside it.

    :return: what it wrote, as a word
    :rtype: str
    """
    assert context.result is not None
    context.result.write(
        {
            "big": pa.table({"a": pa.array([value] * rows, pa.int64()), "s": pa.array([f"r{value}"] * rows)}),
            "small": pa.table({"n": [rows]}),
        }
    )
    return "written"


def filter_version(context: JobContext, table: str, column: str, below: int) -> int:
    """write the version's rows whose ``column`` is below ``below`` as the result, through DuckDB.

    :return: the rows written
    :rtype: int
    """
    assert context.result is not None
    connection = _version(context).connect()
    try:
        kept = connection.execute(f'SELECT * FROM "{table}" WHERE "{column}" < {below}').to_arrow_table()
    finally:
        connection.close()
    context.result.write({"kept": kept})
    rows: int = kept.num_rows
    return rows


def write_then_fail(context: JobContext) -> None:
    """write a result, then raise: nobody claims it.

    :raises ValueError: always
    """
    assert context.result is not None
    context.result.write({"t": pa.table({"a": [1, 2, 3]})})
    raise ValueError("after writing")


def die_mid_write(context: JobContext) -> None:
    """die (no clean-up of any kind) half way through writing a result.

    :return: never
    :rtype: None
    """
    assert context.result is not None
    schema = pa.schema([("a", pa.int64())])

    def batches() -> Iterator[pa.RecordBatch]:
        yield pa.record_batch([pa.array(range(1000), pa.int64())], schema=schema)
        os._exit(17)

    context.result.write({"t": pa.RecordBatchReader.from_batches(schema, batches())})


def write_on_signal(context: JobContext, signals: str) -> None:
    """say it started, wait for the go, then write a result: an abandoned run's worker.

    :return: nothing
    :rtype: None
    """
    assert context.result is not None
    where = Path(signals)
    (where / "started").touch()
    _wait_for(where / "go")
    context.result.write({"t": pa.table({"a": [1]})})


def _digest(mapped: MappedTables, table: str) -> dict[str, Any]:
    """what a handed table holds, read from its mapping: its rows, a digest of its first column's values, and the bytes
    this process's Arrow pool allocated meanwhile (none, when every column is the mapped file's bytes).

    :return: ``rows``, ``sha256`` and ``allocated``
    :rtype: dict[str, Any]
    """
    before = pa.total_allocated_bytes()
    held = mapped.tables[table]
    digest = hashlib.sha256()
    for chunk in held.column(0).chunks:
        digest.update(memoryview(chunk.buffers()[1])[: len(chunk) * 8])
    return {"rows": held.num_rows, "sha256": digest.hexdigest(), "allocated": pa.total_allocated_bytes() - before}


def handed_digest(context: JobContext, table: str) -> dict[str, Any]:
    """:func:`_digest` of a table the pool handed the job.

    :return: as :func:`_digest`
    :rtype: dict[str, Any]
    """
    assert context.tables is not None
    return _digest(context.tables, table)


def digest_of_handed(handed: HandedTables, table: str) -> dict[str, Any]:
    """:func:`_digest` of a table handed to a plain executor's job, mapped here.

    :return: as :func:`_digest`
    :rtype: dict[str, Any]
    """
    with handed.mapped() as mapped:
        return _digest(mapped, table)


def hold_handed_on_signal(handed: HandedTables, signals: str) -> int:
    """map a hand-off, say so, and hold it until the test's go: a worker still reading after its hand-off is released.

    :return: the first table's rows
    :rtype: int
    """
    where = Path(signals)
    with handed.mapped() as mapped:
        (where / "mapped").touch()
        _wait_for(where / "go")
        rows: int = next(iter(mapped.tables.values())).num_rows
        return rows


def handed_on_signal(context: JobContext, signals: str) -> int:
    """say the handed tables are mapped, wait for the go, then answer their rows.

    :return: the rows
    :rtype: int
    """
    assert context.tables is not None
    where = Path(signals)
    (where / "started").touch()
    _wait_for(where / "go")
    return sum(table.num_rows for table in context.tables.tables.values())


def die_holding_handed(context: JobContext) -> None:
    """die (no clean-up of any kind) with the handed tables mapped.

    :return: never
    :rtype: None
    """
    assert context.tables is not None
    os._exit(19)


def die_heard_before_its_lock_goes(context: JobContext, root: str) -> None:
    """close every descriptor but the hand-off's own files, the pipe whose closing tells the pool its worker is gone
    among them, then wait to be killed, the hand-off's lock still held: a worker whose death the pool hears before the
    kernel lets its lock go. A dying process's descriptors close in no order the pool can rely on, so this is the
    order the slowest exit can take, held open.

    :param root: the store's directory
    :ptype root: str
    :return: never
    :rtype: None
    """
    assert context.tables is not None
    handoff = Path(root) / "handoffs" / context.tables.name
    files = {(found.st_dev, found.st_ino) for found in (entry.stat() for entry in os.scandir(handoff))}
    listed = [int(name) for name in os.listdir("/dev/fd")]
    kept: list[int] = []
    for descriptor in listed:
        try:
            found = os.fstat(descriptor)
        except OSError:
            # NOSILENT: the listing's own descriptor, closed once it was listed
            continue
        if (found.st_dev, found.st_ino) in files:
            kept.append(descriptor)
    start = 3
    for descriptor in sorted(kept):
        os.closerange(start, descriptor)
        start = descriptor + 1
    os.closerange(start, max(listed) + 1)
    time.sleep(_WAIT_SECONDS)
    os._exit(19)


def handed_rows(context: JobContext) -> int:
    """the rows of every table the job was handed.

    :return: the rows
    :rtype: int
    """
    assert context.tables is not None
    return sum(table.num_rows for table in context.tables.tables.values())


def part_rows(value: int, rows: int) -> pa.Table:
    """one part's rows: ``a`` is ``value`` in every row, a short text beside it.

    :return: the rows
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array([value] * rows, pa.int64()), "s": pa.array([f"r{value}"] * rows)})


def write_parts(context: JobContext, table: str, parts: int, rows: int) -> int:
    """write a result of ``parts`` parts of ``table``, ``rows`` rows each, in one step.

    :return: the parts written
    :rtype: int
    """
    assert context.result is not None
    schema = part_rows(0, 1).schema
    with context.result.writing() as write:
        write(
            {table: Parts(schema=schema, parts={f"p{value + 1}": part_rows(value + 1, rows) for value in range(parts)})}
        )
    return parts


def hammer_version(root: str, version: str, table: str, column: str, signals: str) -> dict[str, Any]:
    """map ``version`` with no lease, again and again until ``signals``/``stop`` exists, and judge each try: a whole
    read (every row's ``column`` summed as the version holds it), a clean miss (``LookupError``), or anything else (a
    torn read). It touches ``signals``/``started`` after its first try and ``signals``/``missed`` at its first miss.

    Run in a process of its own while the serving process evicts the version: what an eviction must never give a
    reader that races it is a part of the version, or an error that is not a miss.

    :return: ``whole`` and ``missed`` counts, the ``sums`` seen, and ``torn``: each other error's text
    :rtype: dict[str, Any]
    """
    whole = 0
    missed = 0
    sums: set[int] = set()
    torn: list[str] = []
    here = Path(signals)
    while not (here / "stop").exists():
        try:
            with MappedVersion(root, version) as mapped:
                sums.add(int(pc.sum(mapped.tables[table].column(column)).as_py()))
                whole += 1
        except LookupError:
            missed += 1
            (here / "missed").touch()
        except Exception as exc:  # noqa: BLE001 -- the test's verdict: any error but a miss is a torn read
            torn.append(f"{type(exc).__name__}: {exc}")
        (here / "started").touch()
    return {"whole": whole, "missed": missed, "sums": sorted(sums), "torn": torn[:5]}


def mapped_identities(context: JobContext) -> dict[str, Any]:
    """this worker's process id, and each file of its mapped version by its identity (its manifest line) with the id of
    the table it is mapped as: the same id across two jobs is the same table object, taken over as it was, never
    mapped again.

    :return: ``pid`` and ``tables``: identity -> table id
    :rtype: dict[str, Any]
    """
    return {
        "pid": os.getpid(),
        "tables": {str(identity): id(table) for identity, table in _version(context).by_inode.items()},
    }


def identities_on_signal(context: JobContext, signals: str) -> dict[str, Any]:
    """:func:`mapped_identities`, after saying it has mapped its version (``signals``/``started``) and waiting for the
    test's go (``signals``/``go``): a worker busy on a version while the next is published.

    :return: as :func:`mapped_identities`
    :rtype: dict[str, Any]
    """
    where = Path(signals)
    (where / "started").touch()
    _wait_for(where / "go")
    return mapped_identities(context)
