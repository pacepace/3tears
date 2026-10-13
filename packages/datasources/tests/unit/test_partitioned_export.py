"""A relation read through ONE export divided by a column, and the wire it needs.

Every part is proven (one grouped fingerprint before and after the export, the warehouse's count,
each part's manifest count, each part's files) before it is handed back, the export is deleted once
read (or once the reader stops), and a hub that predates partitions or grouping sees the request it
knows when neither is asked for.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import SecretStr

from threetears.datasources.drivers.sql_fragments import build_filter
from threetears.datasources.export import DEFAULT_ROLE, ExportConfig, ExportRefusedError, redshift_unload_statement
from threetears.datasources.export_read import ExportNotDeletedError, IncompleteExportError, export_select
from threetears.datasources.partitioned_export import export_partitions
from threetears.datasources.query_client import (
    DatasourceExportRequest,
    DatasourceExportResult,
    DatasourceQueryRequest,
    RelationFingerprintRequest,
    RelationFingerprintResult,
)

_BUCKET = "bl-eng-aibots-reports-export-dev"
_DESTINATION = "enr/x1/candidates"
_PREFIX = f"exports/reports-warehouse/{_DESTINATION}/"
_COLUMNS = ["join_state_code", "race", "votes"]


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()


def _rows(state: str, n: int) -> list[dict[str, Any]]:
    return [{"join_state_code": state, "race": f"r{i}", "votes": i} for i in range(n)]


class _Store:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        yield self.objects[key]


def _export(parts: dict[str, list[list[dict[str, Any]]]]) -> dict[str, bytes]:
    """a partitioned export: each part's files in its own directory, with a verbose manifest."""
    objects: dict[str, bytes] = {}
    entries = []
    for state, files in parts.items():
        for index, rows in enumerate(files):
            key = f"{_PREFIX}join_state_code={state}/000{index}_part_00.parquet"
            objects[key] = _parquet(rows)
            entries.append({"url": f"s3://{_BUCKET}/{key}", "meta": {"record_count": len(rows)}})
    objects[f"{_PREFIX}manifest"] = json.dumps({"entries": entries}).encode()
    return objects


def _fp(n: int, digest: str = "d") -> RelationFingerprintResult:
    return RelationFingerprintResult(row_count=n, digest=f"{digest}{n}")


class _Client:
    def __init__(self, groups: list[dict[str | None, RelationFingerprintResult]], row_count: int) -> None:
        self._groups = groups
        self._row_count = row_count
        self.asked: list[tuple[str, Any]] = []
        self.deleted: list[str] = []

    async def relation_fingerprint_groups(self, datasource_name: str, **kwargs: Any) -> Any:
        self.asked.append(("groups", kwargs))
        return self._groups.pop(0)

    result_fields: dict[str, Any] = {}
    delete_fails: bool = False

    async def export(
        self, datasource_name: str, select: str, *, destination: str, partition_by: str | None = None
    ) -> Any:
        self.asked.append(("export", (select, destination, partition_by)))
        fields = {
            "row_count": self._row_count,
            "bucket": _BUCKET,
            "object_prefix": _PREFIX,
            "manifest_path": f"{_PREFIX}manifest",
        }
        return DatasourceExportResult(**{**fields, **self.result_fields})

    async def delete_export(self, datasource_name: str, *, destination: str) -> int:
        self.deleted.append(destination)
        if self.delete_fails:
            raise RuntimeError("the bucket refused the delete")
        return 3


async def _read(client: _Client, store: _Store, **kwargs: Any) -> list[tuple[str | None, list[dict[str, Any]]]]:
    return [
        item
        async for item in export_partitions(
            client,  # type: ignore[arg-type]
            store,
            "reports-warehouse",
            relation="reporting.candidates",
            columns=_COLUMNS,
            partition_by="join_state_code",
            destination=_DESTINATION,
            bucket=_BUCKET,
            **kwargs,
        )
    ]


@pytest.mark.asyncio
async def test_one_export_hands_back_every_part_whole_in_order_and_is_deleted() -> None:
    groups = {"DE": _fp(2), "TX": _fp(3)}
    client = _Client([groups, dict(groups)], row_count=5)
    store = _Store(_export({"TX": [_rows("TX", 1), _rows("TX", 3)[1:]], "DE": [_rows("DE", 2)]}))

    read = await _read(client, store)

    assert [(part, len(rows)) for part, rows in read] == [("DE", 2), ("TX", 3)]
    assert [kind for kind, _ in client.asked] == ["groups", "export", "groups"]
    assert client.asked[1][1] == (
        "SELECT join_state_code, race, votes FROM reporting.candidates",
        _DESTINATION,
        "join_state_code",
    )
    assert client.deleted == [_DESTINATION]


@pytest.mark.asyncio
async def test_parts_asked_for_are_exported_alone_and_one_now_empty_comes_back_empty() -> None:
    groups = {"TX": _fp(3)}
    client = _Client([groups, dict(groups)], row_count=3)
    store = _Store(_export({"TX": [_rows("TX", 3)]}))

    read = await _read(client, store, parts=["TX", "WY"])

    assert [(part, len(rows)) for part, rows in read] == [("TX", 3), ("WY", 0)]
    assert "WHERE join_state_code IN ('TX', 'WY')" in client.asked[1][1][0]
    assert client.asked[0][1]["where_in"] == {"join_state_code": ["TX", "WY"]}


@pytest.mark.asyncio
async def test_a_part_that_moved_during_the_export_is_refused_before_any_part_is_handed_back() -> None:
    client = _Client([{"DE": _fp(2), "TX": _fp(3)}, {"DE": _fp(2), "TX": _fp(3, "moved")}], row_count=5)
    store = _Store(_export({"TX": [_rows("TX", 3)], "DE": [_rows("DE", 2)]}))

    with pytest.raises(IncompleteExportError, match="changed while they were exported"):
        await _read(client, store)
    assert client.deleted == [_DESTINATION], "a refused export outlived its load"


@pytest.mark.asyncio
async def test_a_manifest_counting_other_than_a_parts_fingerprint_is_refused() -> None:
    groups = {"DE": _fp(2), "TX": _fp(3)}
    client = _Client([groups, dict(groups)], row_count=5)
    store = _Store(_export({"TX": [_rows("TX", 2)], "DE": [_rows("DE", 3)]}))

    with pytest.raises(IncompleteExportError, match="manifest counts"):
        await _read(client, store)


@pytest.mark.asyncio
async def test_a_manifest_naming_a_part_not_asked_for_or_a_file_outside_a_part_is_refused() -> None:
    groups = {"TX": _fp(3)}
    objects = _export({"TX": [_rows("TX", 3)]})
    manifest = json.loads(objects[f"{_PREFIX}manifest"])
    manifest["entries"].append({"url": f"s3://{_BUCKET}/{_PREFIX}0000_part_00.parquet", "meta": {"record_count": 0}})
    objects[f"{_PREFIX}manifest"] = json.dumps(manifest).encode()
    with pytest.raises(IncompleteExportError, match="not a file of a part"):
        await _read(_Client([groups, dict(groups)], row_count=3), _Store(objects))


@pytest.mark.asyncio
async def test_a_reader_that_stops_early_still_has_the_export_deleted() -> None:
    groups = {"DE": _fp(2), "TX": _fp(3)}
    client = _Client([groups, dict(groups)], row_count=5)
    store = _Store(_export({"TX": [_rows("TX", 3)], "DE": [_rows("DE", 2)]}))
    reading = export_partitions(
        client,  # type: ignore[arg-type]
        store,
        "reports-warehouse",
        relation="reporting.candidates",
        columns=_COLUMNS,
        partition_by="join_state_code",
        destination=_DESTINATION,
        bucket=_BUCKET,
    )
    assert (await anext(reading))[0] == "DE"
    await reading.aclose()
    assert client.deleted == [_DESTINATION]


def test_the_unload_is_partitioned_by_the_column_and_keeps_it() -> None:
    config = ExportConfig(bucket=_BUCKET, prefix="exports/reports-warehouse/", iam_role=DEFAULT_ROLE)
    statement, _ = redshift_unload_statement("SELECT 1", config, _DESTINATION, partition_by="join_state_code")
    assert "PARTITION BY (join_state_code) INCLUDE MANIFEST VERBOSE" in statement
    with pytest.raises(ExportRefusedError):
        redshift_unload_statement("SELECT 1", config, _DESTINATION, partition_by="state); DROP TABLE x --")
    unpartitioned, _ = redshift_unload_statement("SELECT 1", config, _DESTINATION)
    assert "PARTITION" not in unpartitioned


def test_a_set_filter_binds_every_value_and_an_empty_set_keeps_nothing() -> None:
    assert build_filter({"a": "1"}, {"s": ["DE", "TX"]}) == (" WHERE a = $1 AND s IN ($2, $3)", ["1", "DE", "TX"])
    assert build_filter(None, {"s": []}) == (" WHERE 1 = 0", [])
    assert export_select("r.t", ["a"], where_in={"s": ["O'Hare"]}) == "SELECT a FROM r.t WHERE s IN ('O''Hare')"


def test_a_request_that_asks_neither_grouping_nor_partitions_is_what_an_older_hub_knows() -> None:
    """an older hub refuses unknown fields; a plain fingerprint or export must not carry them."""
    plain = DatasourceQueryRequest(
        correlation_id="01a12133-45b2-741e-b410-8b29c0b367c5",
        identity_token=SecretStr("t"),
        fingerprint=RelationFingerprintRequest(relation="r.t", key_columns=["a"]),
    )
    wire = json.loads(plain.model_dump_json())
    assert set(wire["fingerprint"]) == {"relation", "key_columns", "where"}
    export = DatasourceQueryRequest(
        correlation_id="01a12133-45b2-741e-b410-8b29c0b367c5",
        identity_token=SecretStr("t"),
        export=DatasourceExportRequest(select="SELECT 1", destination="d"),
    )
    assert set(json.loads(export.model_dump_json())["export"]) == {"select", "destination"}
    grouped = RelationFingerprintRequest(relation="r.t", key_columns=["a"], group_by="s", where_in={"s": ["DE"]})
    assert json.loads(grouped.model_dump_json())["group_by"] == "s"
    with pytest.raises(ValueError):
        RelationFingerprintRequest(relation="r.t", key_columns=["a"], group_by="s; DROP")


def _proof_case(**change: Any) -> tuple[_Client, _Store]:
    groups = {"DE": _fp(2), "TX": _fp(3)}
    client = _Client([groups, dict(groups)], row_count=5)
    parts = change.pop("parts", {"TX": [_rows("TX", 3)], "DE": [_rows("DE", 2)]})
    client.result_fields = change
    return client, _Store(_export(parts))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"object_prefix": "exports/reports-warehouse/elsewhere/"}, "not the destination"),
        ({"row_count": 6}, "exported 6 rows, the parts hold 5"),
        ({"bucket": "another-bucket"}, "not this reader's"),
        ({"manifest_path": f"{_PREFIX}other-manifest"}, "not this reader's"),
        ({"parts": {"TX": [_rows("TX", 3)], "DE": [_rows("DE", 2)], "VA": [[]]}}, "parts no fingerprint counted"),
        ({"parts": {"TX": [_rows("TX", 3)], "DE": [_rows("DE", 1)]}}, "manifest counts 1 rows"),
    ],
)
async def test_each_proof_refuses_the_export_and_it_is_deleted_all_the_same(
    change: dict[str, Any], match: str, caplog: pytest.LogCaptureFixture
) -> None:
    client, store = _proof_case(**change)
    with caplog.at_level("WARNING"), pytest.raises(IncompleteExportError, match=match):
        await _read(client, store)
    assert client.deleted == [_DESTINATION], "a refused export outlived its load"
    assert "refused" in caplog.text, "the refusal was not logged for an operator"


async def test_files_holding_fewer_rows_than_their_part_are_refused_and_deleted() -> None:
    groups = {"DE": _fp(2), "TX": _fp(3)}
    client = _Client([groups, dict(groups)], row_count=5)
    objects = _export({"TX": [_rows("TX", 3)], "DE": [_rows("DE", 2)]})
    manifest = json.loads(objects[f"{_PREFIX}manifest"])
    for entry in manifest["entries"]:
        entry["meta"].pop("record_count")  # a manifest that does not count leaves the files to prove it
    objects[f"{_PREFIX}manifest"] = json.dumps(manifest).encode()
    objects[f"{_PREFIX}join_state_code=TX/0000_part_00.parquet"] = _parquet(_rows("TX", 2))
    with pytest.raises(IncompleteExportError, match="files hold 2 rows, the part holds 3"):
        await _read(client, _Store(objects))
    assert client.deleted == [_DESTINATION]


async def test_a_delete_that_fails_while_a_refusal_unwinds_does_not_mask_it() -> None:
    client, store = _proof_case(row_count=6)
    client.delete_fails = True
    with pytest.raises(IncompleteExportError, match="exported 6 rows"):
        await _read(client, store)


async def test_a_delete_that_fails_after_a_whole_read_is_its_own_failure() -> None:
    client, store = _proof_case()
    client.delete_fails = True
    with pytest.raises(ExportNotDeletedError, match="could not be deleted"):
        await _read(client, store)


async def test_a_failure_after_the_export_before_any_proof_still_deletes_it() -> None:
    groups = {"DE": _fp(2)}
    client = _Client([groups], row_count=2)  # the second grouped fingerprint finds nothing to answer
    with pytest.raises(IndexError):
        await _read(client, _Store(_export({"DE": [_rows("DE", 2)]})))
    assert client.deleted == [_DESTINATION]


@pytest.mark.parametrize(
    "build",
    [
        lambda: RelationFingerprintRequest(
            relation="r.t", key_columns=["a"], group_by="s", where_in={"s; DROP": ["x"]}
        ),
        lambda: RelationFingerprintRequest(relation="r.t", key_columns=["a"], where_in={"s": ["x"]}),
        lambda: export_select("r.t", ["a"], where_in={"s) OR (1=1": ["x"]}),
        lambda: DatasourceExportRequest(select="SELECT 1", destination="d", partition_by="s) INCLUDE --"),
    ],
)
def test_an_interpolated_identifier_that_is_not_one_is_refused(build: Any) -> None:
    with pytest.raises(ValueError):
        build()
