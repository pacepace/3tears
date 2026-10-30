"""Jobs the shared Arrow tests run in spawned worker processes: module functions, so a worker imports them by name.

Each takes the job's context first, as :class:`~threetears.core.shared_arrow.SharedArrowPool` hands it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.shared_arrow import JobContext, MappedVersion

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


def _mapping_of(path: str) -> dict[str, int]:
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
    return {"pid": os.getpid(), "sum": total, **_mapping_of(os.path.realpath(path))}


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
    return kept.num_rows


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
