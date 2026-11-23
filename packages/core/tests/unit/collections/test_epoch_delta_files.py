"""An epoch record names its commit's delta file (``design-pod-delta-files.md``, decision 4).

- the record carries the file's name and reads it back; one naming none is stored byte for byte as before, and a
  reader built before the field reads a new record as before;
- a name that is not ``<table>/<write id>.arrow`` is refused, and a record of a full run of rows still fits its bound;
- a commit of a table that opts in names its file in the record this process writes, and on the ``l3.tx.commit`` a
  pod sends the broker; a table not opted in, and a commit that did not land, name none.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from threetears.nats import Subject

from threetears.core.backends.broker_generation import DELTA_FILES_FIELD, naming_delta_files
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.epoch_change import (
    ChangedRow,
    advance_recording,
    changed_row_runs,
    check_delta_file,
    epoch_change_body,
    read_epoch_change_body,
)
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.l1_tables import TEST_L1_TABLE

from packages.core.tests.durable_support import ConnOnlyPool

pytest.importorskip("pyarrow")

_FILE = "results/0123456789abcdef0123456789abcdef.arrow"


class _OlderBody(BaseModel):
    """the stored body as a reader built before the delta file reads it."""

    table: str
    whole_table: bool = False
    keys_unknown: bool = False
    rows: list[ChangedRow] = []
    database: str | None = None


def test_a_record_names_its_delta_file_and_reads_it_back() -> None:
    """Flips if the stored record drops the delta file, or a record naming none stops being stored as before."""
    body = epoch_change_body("results", [ChangedRow(ids=["a"])], delta_file=_FILE)
    assert read_epoch_change_body(body, "results", "i:1").delta_file == _FILE
    unnamed = epoch_change_body("results", [ChangedRow(ids=["a"])])
    assert "delta_file" not in json.loads(unnamed)
    assert read_epoch_change_body(unnamed, "results", "i:2").delta_file is None


def test_a_reader_built_before_the_field_reads_a_new_record_as_before() -> None:
    """Flips if a record naming a delta file is one an older reader (a replica not yet upgraded) cannot read."""
    body = epoch_change_body("results", [ChangedRow(ids=["a"])], database="platform", delta_file=_FILE)
    older = _OlderBody.model_validate_json(body)
    assert [row.ids for row in older.rows] == [["a"]] and older.database == "platform"


@pytest.mark.parametrize(
    "name", ["../results/x.arrow", "/etc/x.arrow", "results/x.parquet", "results/a/x.arrow", "results/" + "x" * 300]
)
def test_a_name_that_is_not_a_delta_file_is_refused(name: str) -> None:
    """Flips if a record names a file outside the volume's delta files, or one long enough to break its bound."""
    with pytest.raises(ValueError, match="delta file"):
        epoch_change_body("results", [ChangedRow(ids=["a"])], delta_file=name)
    # a record naming one reads as malformed, as any record breaking its form does: the follower is behind, never misled
    with pytest.raises(GenerationUnavailableError, match="delta file"):
        read_epoch_change_body(
            json.dumps({"table": "results", "rows": [{"ids": ["a"]}], "delta_file": name}).encode(), "results", "i:1"
        )


@pytest.mark.parametrize(
    "name",
    [
        _FILE,
        None,
        "../results/x.arrow",
        "/etc/x.arrow",
        "results/x.parquet",
        "results/a/x.arrow",
        "results/" + "x" * 300,
    ],
)
def test_the_public_check_accepts_and_refuses_the_names_the_record_does(name: str | None) -> None:
    """Flips if the public delta file check (the one a broker calls on a name a pod sent) accepts a name a record
    refuses, or refuses one a record stores."""
    try:
        epoch_change_body("results", [ChangedRow(ids=["a"])], delta_file=name)
    except ValueError as record_refusal:
        with pytest.raises(ValueError, match="delta file") as check_refusal:
            check_delta_file(name)
        assert str(check_refusal.value) == str(record_refusal)
    else:
        check_delta_file(name)


def test_a_full_run_of_rows_still_fits_its_record_with_the_longest_name() -> None:
    """Flips if naming the delta file pushes a record of a run ``changed_row_runs`` filled past the record bound."""
    table = "t" * 63
    rows = [ChangedRow(ids=[f"{i:06d}" + "k" * 200]) for i in range(3_000)]
    runs = changed_row_runs(table, rows)
    assert len(runs) > 1
    name = f"{table}/{'w' * 32}.arrow"
    for run in runs:
        epoch_change_body(table, run, database="platform", delta_file=name)


class _Source:
    """a generation source that records each advance's change, the delta file included."""

    def __init__(self) -> None:
        self.files: list[tuple[str, str | None]] = []

    async def current(self, table_name: str) -> str:
        return f"inc:{len(self.files)}"

    async def advance(self, table_name: str) -> str:
        return f"inc:{len(self.files) + 1}"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        partition: Mapping[str, str] | None = None,
        noted_between: tuple[datetime, datetime] | None = None,
        prefixes: Sequence[Mapping[str, str]] | None = None,
        reason: str | None = None,
        delta_file: str | None = None,
    ) -> str:
        self.files.append((table_name, delta_file))
        return f"inc:{len(self.files)}"


class _Older(_Source):
    """a source built before the delta file: it takes none."""

    async def advance_with_change(  # type: ignore[override]
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str:
        self.files.append((table_name, None))
        return f"inc:{len(self.files)}"


async def test_a_source_built_before_the_field_is_handed_none() -> None:
    """Flips if an advance hands a delta file to a source that does not take one, failing every write through it."""
    source = _Older()
    await advance_recording(source, "results", [ChangedRow(ids=["a"])], delta_file=_FILE)
    assert source.files == [("results", None)]


def _schema(name: str, *, opted_in: bool) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key=("office_key", "geo_id"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("geo_id", STRING_TYPE),
            Column("votes", INT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        l1_eviction=TEST_L1_TABLE,
        delta_files=opted_in,
    )


class _Row(BaseEntity):
    primary_key_field = "geo_id"


def _collection(registry: CollectionRegistry, name: str, *, opted_in: bool) -> Any:
    base = collection_for_schema(_schema(name, opted_in=opted_in), entity_class=_Row)
    switched = type(f"_{name}", (base,), {"write_generation": WRITE_GENERATION})
    collection = switched(registry, DefaultCoreConfig(), None)
    collection.l3_pool = ConnOnlyPool()
    return collection


class _Conn:
    """a caller's connection whose commit lands, or with ``lost`` raises as one whose reply never came back."""

    def __init__(self, *, lost: bool = False) -> None:
        self.lost = lost

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield
        if self.lost:
            raise ConnectionError("the commit's reply never came back")

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        return "INSERT 0 1"


_ROWS = [{"office_key": "sn_NC1", "geo_id": "00001", "votes": 1}]


async def test_a_commit_names_exactly_its_files_and_a_table_not_opted_in_names_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the record this process writes for a commit of an opted-in table does not name exactly the file the
    commit saved, or a table that does not opt in is named one."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    source, registry = _Source(), CollectionRegistry()
    registry.set_generation_source(source)
    results = _collection(registry, "results", opted_in=True)
    totals = _collection(registry, "totals", opted_in=False)
    conn = _Conn()
    async with CallerTransaction(conn) as transaction:
        await results.save_rows(_ROWS, conn=conn)
        await totals.save_rows(_ROWS, conn=conn)
    assert sorted(source.files) == [("results", transaction.delta_files["results"]), ("totals", None)]
    assert set(transaction.delta_files) == {"results"}


async def test_a_commit_that_did_not_land_names_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if the record of a commit whose outcome is unknown names its file: a copy would apply rows L3 may never
    have taken."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    source, registry = _Source(), CollectionRegistry()
    registry.set_generation_source(source)
    results = _collection(registry, "results", opted_in=True)
    conn = _Conn(lost=True)
    with pytest.raises(ConnectionError):
        async with CallerTransaction(conn):
            await results.save_rows(_ROWS, conn=conn)
    assert [file for _, file in source.files] == [None] * len(source.files)


_TX_ID = str(uuid.UUID("00000000-0000-4000-8000-000000000903"))


class _Broker:
    """answers each request with the next scripted reply, and keeps each request's body."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.sent.append((str(subject), json.loads(payload)))
        return json.dumps(self.replies.pop(0)).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
        )

    def body(self, subject_suffix: str) -> dict[str, Any]:
        (body,) = [body for subject, body in self.sent if subject.endswith(subject_suffix)]
        return body


async def test_a_pods_commit_hands_the_broker_its_files() -> None:
    """Flips if a pod's ``l3.tx.commit`` stops carrying the delta files its commit named, for the broker to write
    into the epoch records, or a rollback or a commit naming none carries the field."""
    broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True})
    async with broker.proxy().transaction():
        pass
    assert DELTA_FILES_FIELD not in broker.body("l3.tx.commit")

    broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True})
    with naming_delta_files({"results": _FILE}):
        async with broker.proxy().transaction():
            pass
    assert broker.body("l3.tx.commit")[DELTA_FILES_FIELD] == {"results": _FILE}

    broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True})
    with pytest.raises(RuntimeError), naming_delta_files({"results": _FILE}):
        async with broker.proxy().transaction():
            raise RuntimeError("the body failed")
    assert DELTA_FILES_FIELD not in broker.body("l3.tx.rollback")
