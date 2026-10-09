"""tests for :mod:`threetears.datasources.export_read` and the client's ``export``.

An export's rows are handed back only once proven: the part held still while it was exported, the
warehouse wrote as many rows as the part holds, and the files the manifest lists (and only those,
under the export's own prefix) hold exactly that many. Every way that can fail is a refusal here.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid7

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel

from threetears.datasources.export import ExportRefusedError
from threetears.datasources.export_read import IncompleteExportError, export_part, export_select, read_export
from threetears.datasources.query_client import (
    DatasourceExportResult,
    DatasourceQueryClient,
    DatasourceQueryError,
    DatasourceQueryResponse,
    RelationFingerprintResult,
)
from threetears.nats.subjects import Subject, set_default_namespace

_BUCKET = "bl-eng-aibots-reports-export-dev"
_PREFIX = "exports/enr/r1/VA/"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace("3tears")


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()


class _Store:
    """an export bucket in memory."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.read: list[str] = []

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        self.read.append(key)
        data = self.objects[key]
        for start in range(0, len(data), 7):
            yield data[start : start + 7]


def _export(rows: list[list[dict[str, Any]]], *, counts: bool = True) -> dict[str, bytes]:
    """an export of ``rows`` split into files, with its verbose manifest."""
    objects: dict[str, bytes] = {}
    entries = []
    for index, part in enumerate(rows):
        key = f"{_PREFIX}000{index}_part_00.parquet"
        objects[key] = _parquet(part)
        meta: dict[str, Any] = {"content_length": len(objects[key])}
        if counts:
            meta["record_count"] = len(part)
        entries.append({"url": f"s3://{_BUCKET}/{key}", "meta": meta})
    objects[f"{_PREFIX}manifest"] = json.dumps({"entries": entries}).encode()
    return objects


def _result(row_count: int) -> DatasourceExportResult:
    return DatasourceExportResult(
        row_count=row_count, bucket=_BUCKET, object_prefix=_PREFIX, manifest_path=f"{_PREFIX}manifest"
    )


_ROWS = [
    {"race": "s1", "votes": Decimal("10.500000000000"), "at": datetime(2026, 11, 3, 12, tzinfo=UTC)},
    {"race": "s2", "votes": None, "at": datetime(2026, 11, 3, 13, tzinfo=UTC)},
    {"race": "s3", "votes": Decimal("0E-12"), "at": None},
]


class TestReadExport:
    @pytest.mark.asyncio
    async def test_every_file_the_manifest_lists_is_read_and_values_arrive_typed(self) -> None:
        store = _Store(_export([_ROWS[:2], _ROWS[2:]]))
        rows = await read_export(store, _result(3), bucket=_BUCKET)
        assert rows == _ROWS
        assert store.read[0] == f"{_PREFIX}manifest"
        assert sorted(store.read[1:]) == [f"{_PREFIX}0000_part_00.parquet", f"{_PREFIX}0001_part_00.parquet"]

    @pytest.mark.asyncio
    async def test_files_holding_fewer_rows_than_the_warehouse_wrote_are_refused(self) -> None:
        store = _Store(_export([_ROWS[:2]], counts=False))
        with pytest.raises(IncompleteExportError, match="hold 2 rows"):
            await read_export(store, _result(3), bucket=_BUCKET)

    @pytest.mark.asyncio
    async def test_a_manifest_counting_other_than_the_warehouse_is_refused(self) -> None:
        store = _Store(_export([_ROWS]))
        with pytest.raises(IncompleteExportError, match="manifest counts 3"):
            await read_export(store, _result(4), bucket=_BUCKET)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            f"s3://other-bucket/{_PREFIX}0000_part_00.parquet",
            f"s3://{_BUCKET}/exports/enr/r1/TX/0000_part_00.parquet",
            f"s3://{_BUCKET}/{_PREFIX}sub/0000_part_00.parquet",
            f"s3://{_BUCKET}/{_PREFIX}",
        ],
    )
    async def test_a_manifest_naming_a_file_outside_the_export_is_refused_not_followed(self, url: str) -> None:
        objects = _export([_ROWS])
        objects[f"{_PREFIX}manifest"] = json.dumps({"entries": [{"url": url, "meta": {}}]}).encode()
        store = _Store(objects)
        with pytest.raises(IncompleteExportError, match="not a file of"):
            await read_export(store, _result(3), bucket=_BUCKET)
        assert store.read == [f"{_PREFIX}manifest"]

    @pytest.mark.asyncio
    async def test_an_export_in_another_bucket_is_refused_before_anything_is_read(self) -> None:
        store = _Store({})
        result = DatasourceExportResult(row_count=1, bucket="other", object_prefix=_PREFIX, manifest_path="m")
        with pytest.raises(IncompleteExportError, match="bucket"):
            await read_export(store, result, bucket=_BUCKET)
        assert store.read == []

    @pytest.mark.asyncio
    async def test_an_export_of_no_rows_reads_nothing(self) -> None:
        store = _Store({})
        assert await read_export(store, _result(0), bucket=_BUCKET) == []
        assert store.read == []


class _Client:
    """a datasource client whose fingerprints and export are scripted."""

    def __init__(self, prints: list[RelationFingerprintResult], exported: int) -> None:
        self._prints: Iterator[RelationFingerprintResult] = iter(prints)
        self._exported = exported
        self.calls: list[tuple[str, Any]] = []

    async def relation_fingerprint(self, datasource: str, **kwargs: Any) -> RelationFingerprintResult:
        self.calls.append(("fingerprint", kwargs))
        return next(self._prints)

    async def export(self, datasource: str, select: str, *, destination: str) -> DatasourceExportResult:
        self.calls.append(("export", (select, destination)))
        return _result(self._exported)


def _print(rows: int, digest: str = "d") -> RelationFingerprintResult:
    return RelationFingerprintResult(row_count=rows, digest=digest)


async def _part(client: _Client, store: _Store) -> list[dict[str, Any]]:
    return await export_part(
        client,  # type: ignore[arg-type]
        store,
        "warehouse",
        relation="reporting_prod.results",
        columns=["race", "votes", "at"],
        where={"state": "VA"},
        destination="enr/r1/VA",
        bucket=_BUCKET,
    )


class TestExportPart:
    @pytest.mark.asyncio
    async def test_a_part_that_held_still_and_counts_true_is_returned(self) -> None:
        client = _Client([_print(3), _print(3)], exported=3)
        rows = await _part(client, _Store(_export([_ROWS])))
        assert rows == _ROWS
        kinds = [kind for kind, _ in client.calls]
        assert kinds == ["fingerprint", "export", "fingerprint"]
        fingerprint = client.calls[0][1]
        assert fingerprint == {
            "relation": "reporting_prod.results",
            "key": ["race", "votes", "at"],
            "where": {"state": "VA"},
        }
        assert client.calls[1][1] == (
            "SELECT race, votes, at FROM reporting_prod.results WHERE state = 'VA'",
            "enr/r1/VA",
        )

    @pytest.mark.asyncio
    async def test_a_part_that_moved_during_the_export_is_refused(self) -> None:
        client = _Client([_print(3, "a"), _print(3, "b")], exported=3)
        with pytest.raises(IncompleteExportError, match="changed while it was exported"):
            await _part(client, _Store(_export([_ROWS])))

    @pytest.mark.asyncio
    async def test_an_export_counting_other_than_the_fingerprint_is_refused(self) -> None:
        client = _Client([_print(4), _print(4)], exported=3)
        store = _Store(_export([_ROWS]))
        with pytest.raises(IncompleteExportError, match="exported 3 rows, the relation holds 4"):
            await _part(client, store)
        assert store.read == [], "rows were read from an export already known to be short"


class TestExportSelect:
    def test_values_are_literals_and_names_must_be_identifiers(self) -> None:
        assert export_select("s.t", ["a"], {"state": "O'X"}) == "SELECT a FROM s.t WHERE state = 'O''X'"
        assert export_select("t", ["a", "b"]) == "SELECT a, b FROM t"
        with pytest.raises(ValueError):
            export_select("s.t; DROP TABLE x", ["a"])
        with pytest.raises(ValueError):
            export_select("s.t", ["a, (SELECT 1)"])
        with pytest.raises(ExportRefusedError):
            export_select("s.t", ["a"], {"state": "VA\\"})


# parity-exempt: NatsClient subset for the export client unit test; the client reaches NATS through the typed request form alone
class _FakeNats:
    def __init__(self, reply: DatasourceQueryResponse) -> None:
        self._reply = reply
        self.messages: list[BaseModel] = []

    async def request(
        self, *, subject: Subject, message: BaseModel, response_type: type[BaseModel], timeout: timedelta
    ) -> BaseModel:
        self.messages.append(message)
        return self._reply


class TestClientExport:
    @pytest.mark.asyncio
    async def test_an_export_is_asked_on_the_datasource_subject_and_answered(self) -> None:
        answer = _result(3)
        fake = _FakeNats(DatasourceQueryResponse(success=True, export=answer, correlation_id=uuid7()))
        client = DatasourceQueryClient(fake, identity_token=lambda: "tok")  # type: ignore[arg-type]
        assert await client.export("warehouse", "SELECT 1", destination="enr/x") == answer
        sent = json.loads(fake.messages[0].model_dump_json())
        assert sent["export"] == {"select": "SELECT 1", "destination": "enr/x"}
        assert "query" not in sent or sent["query"] is None

    @pytest.mark.asyncio
    async def test_a_refusal_carries_the_hubs_code(self) -> None:
        fake = _FakeNats(DatasourceQueryResponse(success=False, error_code="EXPORT_NOT_GRANTED", error_message="no"))
        client = DatasourceQueryClient(fake, identity_token=lambda: "tok")  # type: ignore[arg-type]
        with pytest.raises(DatasourceQueryError) as raised:
            await client.export("warehouse", "SELECT 1", destination="enr/x")
        assert raised.value.error_code == "EXPORT_NOT_GRANTED"

    @pytest.mark.asyncio
    async def test_a_success_with_no_export_is_not_taken_as_an_empty_one(self) -> None:
        fake = _FakeNats(DatasourceQueryResponse(success=True, correlation_id=uuid7()))
        client = DatasourceQueryClient(fake, identity_token=lambda: "tok")  # type: ignore[arg-type]
        with pytest.raises(DatasourceQueryError) as raised:
            await client.export("warehouse", "SELECT 1", destination="enr/x")
        assert raised.value.error_code == "MALFORMED_RESPONSE"
