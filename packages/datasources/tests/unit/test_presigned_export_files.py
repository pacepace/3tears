"""tests for handing a warehouse export to L3 as files: each file presigned, with its own row count.

The broker fetches each URL and holds each file to its count, so the count must be the manifest's
own for that file, and the files must be exactly the export's: in the caller's bucket, under the
export's prefix, the manifest the prefix's own, the counts adding up to the rows the warehouse wrote.
A partitioned export's files come grouped by part, the part's value read from its directory.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from threetears.core.backends.protocol import LOAD_FILES_MAX_WAIT_SECONDS, LoadFile
from threetears.datasources.export_read import (
    DEFAULT_PRESIGN_SECONDS,
    IncompleteExportError,
    presigned_export_files,
)
from threetears.datasources.partitioned_export import HIVE_NULL_PARTITION, presigned_partition_files
from threetears.datasources.query_client import DatasourceExportResult

_BUCKET = "bl-eng-aibots-reports-export-dev"
_PREFIX = "exports/enr/lsf-1/"


class _Store:
    """an export bucket in memory that presigns as a recognisable URL."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.presigned: list[tuple[str, int]] = []

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        yield self.objects[key]

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        self.presigned.append((key, expires_in))
        return f"https://{_BUCKET}.s3.amazonaws.com/{key}?X-Amz-Expires={expires_in}"


def _manifest(entries: list[tuple[str, dict[str, Any]]]) -> dict[str, bytes]:
    return {
        f"{_PREFIX}manifest": json.dumps(
            {"entries": [{"url": f"s3://{_BUCKET}/{_PREFIX}{name}", "meta": meta} for name, meta in entries]}
        ).encode()
    }


def _result(row_count: int, **fields: Any) -> DatasourceExportResult:
    return DatasourceExportResult(
        **{
            "row_count": row_count,
            "bucket": _BUCKET,
            "object_prefix": _PREFIX,
            "manifest_path": f"{_PREFIX}manifest",
            **fields,
        }
    )


def _url(key: str, expires_in: int = DEFAULT_PRESIGN_SECONDS) -> str:
    return f"https://{_BUCKET}.s3.amazonaws.com/{key}?X-Amz-Expires={expires_in}"


def test_a_presigned_file_outlives_the_longest_wait_for_its_load() -> None:
    # the broker reaches a file anywhere inside the pod's wait; a URL that expired first fails the load
    assert DEFAULT_PRESIGN_SECONDS > LOAD_FILES_MAX_WAIT_SECONDS


class TestPresignedExportFiles:
    @pytest.mark.asyncio
    async def test_each_file_is_presigned_with_the_rows_the_manifest_counts_for_it(self) -> None:
        store = _Store(
            _manifest([("0000_part_00.parquet", {"record_count": 7}), ("0001_part_00.parquet", {"record_count": 3})])
        )
        files = await presigned_export_files(store, _result(10), bucket=_BUCKET, expires_in=900)
        assert files == [
            LoadFile(url=_url(f"{_PREFIX}0000_part_00.parquet", 900), rows=7),
            LoadFile(url=_url(f"{_PREFIX}0001_part_00.parquet", 900), rows=3),
        ]
        assert [expires for _, expires in store.presigned] == [900, 900]

    @pytest.mark.asyncio
    async def test_an_export_of_no_rows_has_no_files_and_reads_nothing(self) -> None:
        store = _Store({})
        assert await presigned_export_files(store, _result(0), bucket=_BUCKET) == []
        assert store.presigned == []

    @pytest.mark.asyncio
    async def test_a_file_the_manifest_does_not_count_is_refused(self) -> None:
        store = _Store(_manifest([("0000_part_00.parquet", {"content_length": 10})]))
        with pytest.raises(IncompleteExportError, match="counts no rows"):
            await presigned_export_files(store, _result(1), bucket=_BUCKET)
        assert store.presigned == []

    @pytest.mark.asyncio
    async def test_counts_that_differ_from_the_warehouse_s_are_refused(self) -> None:
        store = _Store(_manifest([("0000_part_00.parquet", {"record_count": 4})]))
        with pytest.raises(IncompleteExportError, match="manifest counts 4 rows"):
            await presigned_export_files(store, _result(5), bucket=_BUCKET)
        assert store.presigned == []

    @pytest.mark.asyncio
    async def test_a_manifest_naming_a_file_outside_the_export_is_refused(self) -> None:
        objects = {
            f"{_PREFIX}manifest": json.dumps(
                {"entries": [{"url": f"s3://{_BUCKET}/exports/elsewhere/0000.parquet", "meta": {"record_count": 1}}]}
            ).encode()
        }
        with pytest.raises(IncompleteExportError, match="not a file of"):
            await presigned_export_files(_Store(objects), _result(1), bucket=_BUCKET)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fields", "match"),
        [({"bucket": "other"}, "bucket"), ({"manifest_path": f"{_PREFIX}other"}, "not its prefix")],
    )
    async def test_an_export_elsewhere_is_refused_before_anything_is_read(
        self, fields: dict[str, Any], match: str
    ) -> None:
        store = _Store({})
        with pytest.raises(IncompleteExportError, match=match):
            await presigned_export_files(store, _result(1, **fields), bucket=_BUCKET)


class TestPresignedPartitionFiles:
    @pytest.mark.asyncio
    async def test_files_come_grouped_by_the_part_their_directory_names(self) -> None:
        store = _Store(
            _manifest(
                [
                    ("state=VA/0000_part_00.parquet", {"record_count": 2}),
                    ("state=TX/0000_part_00.parquet", {"record_count": 5}),
                    ("state=VA/0001_part_00.parquet", {"record_count": 1}),
                    (f"state={HIVE_NULL_PARTITION}/0000_part_00.parquet", {"record_count": 4}),
                ]
            )
        )
        parts = await presigned_partition_files(store, _result(12), bucket=_BUCKET, partition_by="state")
        assert parts == {
            "VA": [
                LoadFile(url=_url(f"{_PREFIX}state=VA/0000_part_00.parquet"), rows=2),
                LoadFile(url=_url(f"{_PREFIX}state=VA/0001_part_00.parquet"), rows=1),
            ],
            "TX": [LoadFile(url=_url(f"{_PREFIX}state=TX/0000_part_00.parquet"), rows=5)],
            None: [LoadFile(url=_url(f"{_PREFIX}state={HIVE_NULL_PARTITION}/0000_part_00.parquet"), rows=4)],
        }

    @pytest.mark.asyncio
    async def test_counts_that_differ_from_the_warehouse_s_are_refused(self) -> None:
        store = _Store(_manifest([("state=VA/0000_part_00.parquet", {"record_count": 2})]))
        with pytest.raises(IncompleteExportError, match="manifest counts 2 rows"):
            await presigned_partition_files(store, _result(3), bucket=_BUCKET, partition_by="state")

    @pytest.mark.asyncio
    async def test_a_file_outside_a_part_s_directory_is_refused(self) -> None:
        store = _Store(_manifest([("0000_part_00.parquet", {"record_count": 2})]))
        with pytest.raises(IncompleteExportError, match="not a file of a part"):
            await presigned_partition_files(store, _result(2), bucket=_BUCKET, partition_by="state")

    @pytest.mark.asyncio
    async def test_a_file_the_manifest_does_not_count_is_refused(self) -> None:
        store = _Store(_manifest([("state=VA/0000_part_00.parquet", {})]))
        with pytest.raises(IncompleteExportError, match="counts no rows"):
            await presigned_partition_files(store, _result(2), bucket=_BUCKET, partition_by="state")

    @pytest.mark.asyncio
    async def test_an_export_of_no_rows_has_no_parts(self) -> None:
        assert await presigned_partition_files(_Store({}), _result(0), bucket=_BUCKET, partition_by="state") == {}
