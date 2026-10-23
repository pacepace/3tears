"""A caller's SELECT exported divided by a column, proven against the caller's own counts.

For rows that never change once written: no fingerprint, the caller's per-part counts instead. Every
proof refuses before a row is read and the refused export is deleted; the proven export's parts are
read one at a time, each held to its count, or handed to L3 as presigned files; the caller deletes
it once done.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from threetears.core.backends.protocol import LoadFile
from threetears.datasources.export_read import ExportNotDeletedError, IncompleteExportError
from threetears.datasources.partitioned_export import (
    CountedFile,
    CountedPartitionedExport,
    delete_counted_export,
    export_counted_partitions,
    presigned_counted_partition,
    read_counted_partition,
)
from threetears.datasources.query_client import DatasourceExportResult

_BUCKET = "bl-eng-aibots-reports-export-dev"
_DESTINATION = "lsf/x1/projections"
_PREFIX = f"exports/reports-warehouse/{_DESTINATION}/"
_SELECT = "SELECT join_state_code, run_id, quantity FROM lsf WHERE run_id IN (1, 2)"


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()


def _rows(state: str, n: int) -> list[dict[str, Any]]:
    return [{"join_state_code": state, "run_id": 1, "quantity": f"q{i}"} for i in range(n)]


class _Store:
    """an export bucket in memory that presigns as a recognisable URL."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        yield self.objects[key]

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        return f"https://{_BUCKET}.s3.amazonaws.com/{key}?X-Amz-Expires={expires_in}"


def _export(parts: dict[str, list[list[dict[str, Any]]]], *, counted: bool = True) -> dict[str, bytes]:
    """a partitioned export: each part's files in its own directory, with a manifest."""
    objects: dict[str, bytes] = {}
    entries = []
    for state, files in parts.items():
        for index, rows in enumerate(files):
            key = f"{_PREFIX}join_state_code={state}/000{index}_part_00.parquet"
            objects[key] = _parquet(rows)
            entries.append({"url": f"s3://{_BUCKET}/{key}", "meta": {"record_count": len(rows)} if counted else {}})
    objects[f"{_PREFIX}manifest"] = json.dumps({"entries": entries}).encode()
    return objects


class _Client:
    def __init__(self, row_count: int, **fields: Any) -> None:
        self._fields = {
            "row_count": row_count,
            "bucket": _BUCKET,
            "object_prefix": _PREFIX,
            "manifest_path": f"{_PREFIX}manifest",
            **fields,
        }
        self.asked: list[tuple[str, str, str | None]] = []
        self.deleted: list[str] = []
        self.delete_fails = False

    async def export(
        self, datasource_name: str, select: str, *, destination: str, partition_by: str | None = None
    ) -> DatasourceExportResult:
        self.asked.append((select, destination, partition_by))
        return DatasourceExportResult(**self._fields)

    async def delete_export(self, datasource_name: str, *, destination: str) -> int:
        self.deleted.append(destination)
        if self.delete_fails:
            raise RuntimeError("the bucket refused the delete")
        return 2


async def _counted(client: _Client, store: _Store, counts: dict[str, int]) -> CountedPartitionedExport:
    return await export_counted_partitions(
        client,  # type: ignore[arg-type]
        store,
        "reports-warehouse",
        select=_SELECT,
        partition_by="join_state_code",
        counts=counts,
        destination=_DESTINATION,
        bucket=_BUCKET,
    )


@pytest.mark.asyncio
async def test_the_caller_s_select_is_exported_and_proven_against_its_counts() -> None:
    client = _Client(row_count=5)
    store = _Store(_export({"TX": [_rows("TX", 1), _rows("TX", 2)], "DE": [_rows("DE", 2)]}))

    export = await _counted(client, store, {"TX": 3, "DE": 2, "WY": 0})

    assert client.asked == [(_SELECT, _DESTINATION, "join_state_code")]
    assert export.rows == {"TX": 3, "DE": 2}, "a part counted with none is no part of the export"
    assert export.files["TX"] == (
        CountedFile(path=f"{_PREFIX}join_state_code=TX/0000_part_00.parquet", rows=1),
        CountedFile(path=f"{_PREFIX}join_state_code=TX/0001_part_00.parquet", rows=2),
    )
    assert client.deleted == [], "a proven export is the caller's to delete"
    assert [row["quantity"] for row in await read_counted_partition(store, export, "TX")] == ["q0", "q0", "q1"]
    assert await read_counted_partition(store, export, "WY") == []


@pytest.mark.asyncio
async def test_a_part_s_files_are_presigned_with_their_manifest_counts() -> None:
    store = _Store(_export({"TX": [_rows("TX", 1), _rows("TX", 2)]}))
    export = await _counted(_Client(row_count=3), store, {"TX": 3})

    files = await presigned_counted_partition(store, export, "TX", expires_in=900)

    assert files == [
        LoadFile(
            url=f"https://{_BUCKET}.s3.amazonaws.com/{_PREFIX}join_state_code=TX/0000_part_00.parquet?X-Amz-Expires=900",
            rows=1,
        ),
        LoadFile(
            url=f"https://{_BUCKET}.s3.amazonaws.com/{_PREFIX}join_state_code=TX/0001_part_00.parquet?X-Amz-Expires=900",
            rows=2,
        ),
    ]
    assert await presigned_counted_partition(store, export, "WY") == []


@pytest.mark.asyncio
async def test_a_file_the_manifest_does_not_count_can_be_read_but_not_presigned() -> None:
    store = _Store(_export({"TX": [_rows("TX", 3)]}, counted=False))
    export = await _counted(_Client(row_count=3), store, {"TX": 3})

    assert len(await read_counted_partition(store, export, "TX")) == 3
    with pytest.raises(IncompleteExportError, match="counts no rows"):
        await presigned_counted_partition(store, export, "TX")


@pytest.mark.asyncio
async def test_a_padded_part_value_is_the_counted_part() -> None:
    store = _Store(_export({"TX%20": [_rows("TX", 2)]}))
    export = await _counted(_Client(row_count=2), store, {"TX": 2})
    assert list(export.files) == ["TX"]


@pytest.mark.parametrize(
    ("fields", "parts", "counts", "match"),
    [
        ({"object_prefix": "exports/reports-warehouse/elsewhere/"}, {"TX": [3]}, {"TX": 3}, "not the destination"),
        ({"bucket": "another-bucket"}, {"TX": [3]}, {"TX": 3}, "not this reader's"),
        ({"manifest_path": f"{_PREFIX}other"}, {"TX": [3]}, {"TX": 3}, "not this reader's"),
        ({"row_count": 4}, {"TX": [3]}, {"TX": 3}, "exported 4 rows, the counts add up to 3"),
        ({}, {"TX": [3], "VA": [0]}, {"TX": 3}, "no count named"),
        ({}, {"TX": [2]}, {"TX": 3}, "manifest counts 2 rows, the caller 3"),
        ({"row_count": 5}, {"TX": [3]}, {"TX": 3, "DE": 2}, r"no file of join_state_code \['DE'\]"),
    ],
)
@pytest.mark.asyncio
async def test_each_proof_refuses_and_the_export_is_deleted_before_the_refusal_is_raised(
    fields: dict[str, Any],
    parts: dict[str, list[int]],
    counts: dict[str, int],
    match: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _Client(row_count=fields.pop("row_count", sum(counts.values())), **fields)
    store = _Store(_export({state: [_rows(state, n) for n in files] for state, files in parts.items()}))
    with caplog.at_level("WARNING"), pytest.raises(IncompleteExportError, match=match):
        await _counted(client, store, counts)
    assert client.deleted == [_DESTINATION], "a refused export outlived its read"
    assert "refused" in caplog.text, "the refusal was not logged for an operator"


@pytest.mark.asyncio
async def test_a_delete_that_fails_under_a_refusal_does_not_mask_it() -> None:
    client = _Client(row_count=4)
    client.delete_fails = True
    with pytest.raises(IncompleteExportError, match="exported 4 rows"):
        await _counted(client, _Store(_export({"TX": [_rows("TX", 3)]})), {"TX": 3})


@pytest.mark.asyncio
async def test_a_caller_that_stops_waiting_mid_proof_has_the_export_deleted() -> None:
    client = _Client(row_count=3)
    reading = asyncio.Event()

    class _SlowStore(_Store):
        async def open_read(self, key: str) -> AsyncIterator[bytes]:
            reading.set()
            await asyncio.sleep(60)
            yield self.objects[key]

    task = asyncio.create_task(_counted(client, _SlowStore(_export({"TX": [_rows("TX", 3)]})), {"TX": 3}))
    await reading.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.deleted == [_DESTINATION]


@pytest.mark.asyncio
async def test_files_holding_other_than_the_count_are_refused_when_the_part_is_read() -> None:
    objects = _export({"TX": [_rows("TX", 3)]}, counted=False)
    objects[f"{_PREFIX}join_state_code=TX/0000_part_00.parquet"] = _parquet(_rows("TX", 2))
    store = _Store(objects)
    export = await _counted(_Client(row_count=3), store, {"TX": 3})
    with pytest.raises(IncompleteExportError, match="files hold 2 rows, the caller counted 3"):
        await read_counted_partition(store, export, "TX")


@pytest.mark.asyncio
async def test_the_caller_deletes_a_proven_export_and_a_failed_delete_is_raised_unless_unwinding() -> None:
    client = _Client(row_count=3)
    export = await _counted(client, _Store(_export({"TX": [_rows("TX", 3)]})), {"TX": 3})

    assert await delete_counted_export(client, "reports-warehouse", export) == 2  # type: ignore[arg-type]
    assert client.deleted == [_DESTINATION]

    client.delete_fails = True
    with pytest.raises(ExportNotDeletedError, match="could not be deleted"):
        await delete_counted_export(client, "reports-warehouse", export)  # type: ignore[arg-type]
    unwound = await delete_counted_export(
        client,  # type: ignore[arg-type]
        "reports-warehouse",
        export,
        unwinding=RuntimeError("the load failed"),
    )
    assert unwound is None
