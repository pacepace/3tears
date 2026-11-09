"""An L1 keeps a timestamp key in the one spelling its change record names it in, so an eviction by that text finds it.

A ``timestamptz`` key is named by :func:`~threetears.core.collections.epoch_change.key_text` in every change record,
row message, L2 key and fence key: UTC, to the microsecond (``2026-11-09T10:22:10.000000+00:00``). The SQLite and DuckDB
L1s kept it as ``isoformat()``, which drops ``.000000`` from a whole second and keeps the value's own offset, so an
eviction naming a whole-second key (or one written at another offset) by its record's text missed the row, and the pod
kept serving the stale value. Both L1s live in process memory only, so nothing kept in the old spelling outlives the
process that wrote it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


#: a whole second, where ``isoformat`` drops the fraction the record's text carries
WHOLE = datetime(2026, 11, 9, 10, 22, 10, tzinfo=UTC)
#: a sub-second instant
SUB = datetime(2026, 11, 9, 10, 22, 10, 82658, tzinfo=UTC)
#: a whole second written at another offset, which ``isoformat`` kept as it came
ZONED = datetime(2026, 11, 9, 5, 22, 11, tzinfo=timezone(timedelta(hours=-5)))

INSTANTS = pytest.mark.parametrize("instant", [WHOLE, SUB, ZONED], ids=["whole-second", "sub-second", "other-offset"])


def _metadata() -> MetaData:
    md = MetaData()
    Table("ticks", md, Column("at", DateTime(timezone=True), primary_key=True), Column("v", Integer))
    Table(
        "race_ticks",
        md,
        Column("race", String(64), primary_key=True),
        Column("at", DateTime(timezone=True), primary_key=True),
        Column("v", Integer),
    )
    return md


@pytest.fixture(params=["sqlite", "duckdb"])
def l1(request: pytest.FixtureRequest) -> Iterator[Any]:
    backend: Any = (
        SQLiteBackend(db_name=f"instant_{uuid.uuid4().hex[:8]}") if request.param == "sqlite" else DuckDBBackend()
    )
    backend.initialize(_metadata())
    yield backend
    backend.reset()


@INSTANTS
def test_a_timestamp_key_is_evicted_by_its_record_text(l1: Any, instant: datetime) -> None:
    l1.upsert("ticks", {"at": instant, "v": 1}, "at")
    assert l1.select_by_id("ticks", instant, "at") is not None
    l1.delete_by_id("ticks", key_text(instant), "at")
    assert l1.select_by_id("ticks", instant, "at") is None


@INSTANTS
def test_many_timestamp_keys_are_evicted_by_their_record_text(l1: Any, instant: datetime) -> None:
    later = instant + timedelta(seconds=1)
    l1.upsert_many("ticks", [{"at": instant, "v": 1}, {"at": later, "v": 2}], "at")
    l1.delete_many_by_id("ticks", [key_text(instant), key_text(later)], "at")
    assert l1.select_batch("ticks", [instant, later], "at") == []


@INSTANTS
def test_a_composite_key_with_an_instant_is_evicted_by_its_record_text(l1: Any, instant: datetime) -> None:
    key = ("race", "at")
    l1.upsert("race_ticks", {"race": "r1", "at": instant, "v": 1}, key)
    l1.delete_many_by_id("race_ticks", [("r1", key_text(instant))], key)
    assert l1.select_by_id("race_ticks", ("r1", instant), key) is None


@INSTANTS
def test_a_row_is_read_by_any_spelling_of_its_instant(l1: Any, instant: datetime) -> None:
    l1.upsert("ticks", {"at": instant, "v": 7}, "at")
    for named in (instant, key_text(instant), instant.isoformat(), instant.astimezone(UTC).isoformat()):
        row = l1.select_by_id("ticks", named, "at")
        assert row is not None and row["v"] == 7 and row["at"] == instant, named


@INSTANTS
def test_an_instant_is_stored_as_its_record_names_it(l1: Any, instant: datetime) -> None:
    l1.upsert("ticks", {"at": instant, "v": 1}, "at")
    assert [row["at"] for row in l1.execute_query('SELECT "at" FROM ticks')] == [key_text(instant)]


@INSTANTS
def test_a_duckdb_row_from_arrow_text_is_evicted_by_its_record_text(instant: datetime) -> None:
    # the L3 rail answers a timestamp as ISO text ("...Z"); the copy kept that text as it came
    l1 = DuckDBBackend()
    l1.initialize(_metadata())
    try:
        rail_text = instant.astimezone(UTC).isoformat().replace("+00:00", "Z")
        l1.stage_rows(STAGE_PREFIX + "rows", "ticks", pa.table({"at": [rail_text], "v": [1]}), ["at"])
        l1.apply_staged([StagedChange(table="ticks", key=("at",), rows=STAGE_PREFIX + "rows")])
        assert l1.select_by_id("ticks", instant, "at") is not None
        l1.delete_by_id("ticks", key_text(instant), "at")
        assert l1.select_by_id("ticks", instant, "at") is None
    finally:
        l1.reset()


@INSTANTS
def test_a_duckdb_change_naming_its_keys_by_text_removes_their_rows(instant: datetime) -> None:
    l1 = DuckDBBackend()
    l1.initialize(_metadata())
    try:
        l1.upsert("ticks", {"at": instant, "v": 1}, "at")
        l1.stage_keys(STAGE_PREFIX + "keys", "ticks", ["at"], pa.table({"at": [key_text(instant)]}))
        l1.apply_staged([StagedChange(table="ticks", key=("at",), keys=STAGE_PREFIX + "keys")])
        assert l1.select_by_id("ticks", instant, "at") is None
    finally:
        l1.reset()


@INSTANTS
def test_a_duckdb_row_loaded_from_parquet_is_evicted_by_its_record_text(tmp_path: Path, instant: datetime) -> None:
    path = tmp_path / "ticks.parquet"
    pq.write_table(pa.table({"at": pa.array([instant], pa.timestamp("us", tz="UTC")), "v": [1]}), path)
    l1 = DuckDBBackend()
    l1.initialize(_metadata())
    try:
        assert l1.load_parquet("ticks", path) == 1
        assert l1.select_by_id("ticks", instant, "at") is not None
        l1.delete_by_id("ticks", key_text(instant), "at")
        assert l1.select_by_id("ticks", instant, "at") is None
    finally:
        l1.reset()


class TickEntity(BaseEntity):
    primary_key_field = "at"


class TickCollection(BaseCollection[TickEntity]):
    """a collection keyed by an instant, on a SQLite L1."""

    primary_key_column: str | tuple[str, ...] = "at"

    @property
    def table_name(self) -> str:
        return "ticks"

    @property
    def entity_class(self) -> type[TickEntity]:
        return TickEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


@INSTANTS
async def test_a_row_message_naming_a_timestamp_key_evicts_it_from_l1(instant: datetime) -> None:
    l1 = SQLiteBackend(db_name=f"instant_coll_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    bucket = AsyncMock()
    bucket.delete = AsyncMock(return_value=True)
    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    collection = TickCollection(registry, DefaultCoreConfig(), nats_client=nats)
    subscribers: list[Any] = []

    async def _subscribe_typed(*, subject: Any, cb: Any, message_type: Any, **kwargs: Any) -> None:  # noqa: ARG001
        subscribers.append(cb)

    nats.subscribe_typed = AsyncMock(side_effect=_subscribe_typed)
    try:
        collection.write_to_cache_sync({"at": instant, "v": 1})
        assert collection.get_row_sync(instant) is not None
        await registry.start_invalidation_listener(nats)
        await subscribers[0](CacheInvalidationMessage(table="ticks", ids=[key_text(instant)]))
        assert collection.get_row_sync(instant) is None
    finally:
        l1.reset()
