"""every collection L2 entry carries its shape tag, and a read ignores an entry that is not its shape.

A release that changes a table's shape used to need the shared collections bucket cleared at its first start, or a new
pod decoded an older pod's entry as its own. Each entry now opens with its collection's shape tag
(:mod:`threetears.core.collections.l2_shape`). What is pinned:

- an entry with no tag (every entry written before 0.95.0) or another shape's tag is ignored: the row is read from L3,
  the entry replaced at the revision read, and the mismatch counted on
  ``threetears_collection_l2_shape_mismatches_total{table}``;
- an entry of the collection's own shape is served, and L3 is not read;
- a stale entry L3 holds no row for is removed;
- the tag moves with a declared column's name, type or order, the key, the row epoch and the move columns, and not with
  an index, nullability, a default, a foreign key, tenancy or eviction;
- a collection with no TableSchema tags by its key, its instants, its declared columns and its ``l2_shape_version``;
- where L2 is the table's record (``l2_first``), an entry of another shape is still the committed row, never "no row".
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, ClassVar

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import Column as SaColumn
from sqlalchemy import DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.l2_first import L2_FIRST
from threetears.core.collections.l2_shape import L2ShapeMismatch
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    INT_TYPE,
    STRING_TYPE,
    Column,
    IndexDef,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.entities.base import BaseEntity
from threetears.core.tenancy import no_tenant
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

_SCOPE = "shape-principal"
_TABLE = "shape_profiles"
_ID = "p-1"
_KEY = f"{_SCOPE}.{_TABLE}.{_ID}"
_METRIC = "threetears_collection_l2_shape_mismatches_total"


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("id", String(255), primary_key=True),
        SaColumn("name", String(255)),
        SaColumn("date_created", DateTime(timezone=True)),
        SaColumn("date_updated", DateTime(timezone=True)),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Profiles(BaseCollection[_Row]):
    """a three-tier collection with no TableSchema: its tag is what it declares."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows
        self.fetches = 0
        super().__init__(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self.fetches += 1
        row = self.rows.get(str(entity_id))
        return None if row is None else dict(row)

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self.rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _ProfilesV2(_Profiles):
    """the same table, its own codec changed: the class raises its version."""

    l2_shape_version: ClassVar[int] = 1


class _ProfilesWithColumns(_Profiles):
    """the same table, its columns named for the tag."""

    l2_shape_columns: ClassVar[tuple[tuple[str, str], ...]] = (("id", "text"), ("name", "text"))


class _ProfilesWithOtherColumns(_Profiles):
    """the same table, a column retyped."""

    l2_shape_columns: ClassVar[tuple[tuple[str, str], ...]] = (("id", "text"), ("name", "integer"))


class _L2FirstProfiles(_Profiles):
    """the same table, committed L2-first: L2 is its record."""

    l2_first = L2_FIRST


def _registry(nats: FakeNatsClient) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"shape_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return registry


def _replica[C: _Profiles](cls: type[C], nats: FakeNatsClient, rows: dict[str, dict[str, Any]]) -> C:
    return cls(_registry(nats), rows)


async def _bucket(nats: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await nats.kv_bucket(name="collections")
    return bucket


def _mismatches() -> float:
    return REGISTRY.get_sample_value(_METRIC, {"table": _TABLE}) or 0.0


def _untagged(row: dict[str, Any]) -> bytes:
    """an entry as every release before 0.95.0 wrote it: the codec's bytes, no tag."""
    return json.dumps(row).encode("utf-8")


def _other_tag(row: dict[str, Any]) -> bytes:
    """an entry another shape wrote: a well-formed tag that is not this collection's."""
    return b"\x00shape:" + b"0123456789abcdef" + json.dumps(row).encode("utf-8")


class TestAnEntryOfAnotherShapeIsIgnored:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("entry", [_untagged, _other_tag], ids=["no tag", "old tag"])
    async def test_it_is_read_from_l3_and_the_entry_replaced(self, entry: Any) -> None:
        nats = FakeNatsClient()
        reader = _replica(_Profiles, nats, {_ID: {"id": _ID, "name": "from l3"}})
        bucket = await _bucket(nats)
        await bucket.put(key=_KEY, value=entry({"id": _ID, "name": "from an older release"}))
        before = _mismatches()

        row = await reader.ensure(_ID)

        assert row is not None and row["name"] == "from l3", "a read answered with an entry of another shape"
        assert reader.fetches == 1
        assert _mismatches() == before + 1, "the ignored entry was not counted"
        raw = await bucket.get(key=_KEY)
        assert raw is not None
        refilled = reader.decode_l2_entry(raw)
        assert refilled is not None and refilled["name"] == "from l3", "the entry was not replaced from L3"

    @pytest.mark.asyncio
    async def test_an_entry_l3_holds_no_row_for_is_removed(self) -> None:
        nats = FakeNatsClient()
        reader = _replica(_Profiles, nats, {})
        bucket = await _bucket(nats)
        await bucket.put(key=_KEY, value=_untagged({"id": _ID, "name": "deleted since"}))

        assert await reader.ensure(_ID) is None
        assert await bucket.get(key=_KEY) is None, "a stale entry of another shape was left in L2"

    @pytest.mark.asyncio
    async def test_the_decoder_raises_it_as_a_corrupt_entry_never_as_a_row(self) -> None:
        reader = _replica(_Profiles, FakeNatsClient(), {})
        with pytest.raises(L2ShapeMismatch):
            reader.decode_l2_entry(_untagged({"id": _ID, "name": "x"}))


class TestAnEntryOfItsOwnShapeIsServed:
    @pytest.mark.asyncio
    async def test_a_written_entry_is_read_without_l3(self) -> None:
        nats = FakeNatsClient()
        rows: dict[str, dict[str, Any]] = {_ID: {"id": _ID, "name": "from l3"}}
        reader = _replica(_Profiles, nats, rows)
        bucket = await _bucket(nats)
        await bucket.put(key=_KEY, value=reader.encode_l2_entry({"id": _ID, "name": "from l2"}))
        before = _mismatches()

        row = await reader.ensure(_ID)

        assert row is not None and row["name"] == "from l2"
        assert reader.fetches == 0, "an entry of the collection's own shape sent the read to L3"
        assert _mismatches() == before

    @pytest.mark.asyncio
    async def test_a_save_writes_the_tag_another_replica_reads(self) -> None:
        nats = FakeNatsClient()
        rows: dict[str, dict[str, Any]] = {}
        writer = _replica(_Profiles, nats, rows)
        entity = writer.entity_class({"id": _ID, "name": "saved"}, is_new=True, collection=writer)
        await writer.save_entity(entity)
        rows[_ID] = {"id": _ID, "name": "L3 copy"}  # L3 now differs, so a hit is visibly L2's

        reader = _replica(_Profiles, nats, rows)
        row = await reader.ensure(_ID)

        assert row is not None and row["name"] == "saved"
        assert reader.fetches == 0


class TestATagWithoutASchemaIsWhatTheClassDeclares:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("writer_cls", "reader_cls"),
        [
            (_Profiles, _ProfilesV2),
            (_ProfilesWithColumns, _ProfilesWithOtherColumns),
            (_Profiles, _ProfilesWithColumns),
        ],
        ids=["l2_shape_version raised", "a declared column retyped", "columns declared"],
    )
    async def test_a_changed_declaration_ignores_the_older_entry(
        self, writer_cls: type[_Profiles], reader_cls: type[_Profiles]
    ) -> None:
        nats = FakeNatsClient()
        rows: dict[str, dict[str, Any]] = {_ID: {"id": _ID, "name": "from l3"}}
        writer = _replica(writer_cls, nats, rows)
        await (await _bucket(nats)).put(key=_KEY, value=writer.encode_l2_entry({"id": _ID, "name": "older shape"}))

        reader = _replica(reader_cls, nats, rows)
        row = await reader.ensure(_ID)

        assert row is not None and row["name"] == "from l3"
        assert reader.fetches == 1

    def test_the_same_declaration_reads_the_same_tag(self) -> None:
        nats = FakeNatsClient()
        one, two = _replica(_Profiles, nats, {}), _replica(_Profiles, nats, {})
        row = {"id": _ID, "name": "x"}
        assert one.encode_l2_entry(row) == two.encode_l2_entry(row)
        assert two.decode_l2_entry(one.encode_l2_entry(row)) == row


class TestWhereL2IsTheRecord:
    def test_an_entry_of_another_shape_is_still_the_committed_row(self) -> None:
        # an L2-first table's entry exists nowhere else until the saver carries it to L3: read as "no row", the saver
        # would delete the row from L3. It is counted, and decoded as it is.
        record = _replica(_L2FirstProfiles, FakeNatsClient(), {})
        before = _mismatches()
        for raw in (_untagged({"id": _ID, "name": "committed"}), _other_tag({"id": _ID, "name": "committed"})):
            row = record.decode_l2_entry(raw)
            assert row is not None and row["name"] == "committed"
        assert _mismatches() == before + 2


def _schema(**changes: Any) -> TableSchema:
    base = TableSchema(
        name="shape_widgets",
        primary_key="id",
        columns=[
            Column("id", STRING_TYPE),
            Column("name", STRING_TYPE, nullable=True),
            Column("count", INT_TYPE, nullable=True),
        ],
    )
    return replace(base, **changes)


class TestTheSchemaTagCoversWhatChangesAnEntry:
    def test_it_is_stable(self) -> None:
        assert _schema().l2_shape == _schema().l2_shape
        assert len(_schema().l2_shape) == 8

    @pytest.mark.parametrize(
        "changes",
        [
            {"columns": [Column("id", STRING_TYPE), Column("label", STRING_TYPE), Column("count", INT_TYPE)]},
            {"columns": [Column("id", STRING_TYPE), Column("name", INT_TYPE), Column("count", INT_TYPE)]},
            {"columns": [Column("id", STRING_TYPE), Column("count", INT_TYPE), Column("name", STRING_TYPE)]},
            {"columns": [Column("id", STRING_TYPE), Column("name", STRING_TYPE)]},
            {"primary_key": ("id", "name")},
            {"row_epoch": ROW_STAMPED},
        ],
        ids=["column renamed", "column retyped", "columns reordered", "column dropped", "key changed", "row epoch"],
    )
    def test_a_declared_shape_change_moves_it(self, changes: dict[str, Any]) -> None:
        assert _schema(**changes).l2_shape != _schema().l2_shape

    @pytest.mark.parametrize(
        "changes",
        [
            {"indexes": (IndexDef(name="ix_shape_widgets_name", columns=("name",)),)},
            {"columns": [Column("id", STRING_TYPE), Column("name", STRING_TYPE), Column("count", INT_TYPE)]},
            {
                "columns": [
                    Column("id", STRING_TYPE),
                    Column("name", STRING_TYPE, nullable=True, server_default="'x'"),
                    Column("count", INT_TYPE, nullable=True),
                ]
            },
            {"tenancy": no_tenant("a test table")},
            {"l1_eviction": TEST_L1_TABLE},
            {"cas_column": "name"},
        ],
        ids=["index", "nullability", "server default", "tenancy", "eviction", "cas column"],
    )
    def test_what_does_not_change_an_entry_leaves_it(self, changes: dict[str, Any]) -> None:
        assert _schema(**changes).l2_shape == _schema().l2_shape
