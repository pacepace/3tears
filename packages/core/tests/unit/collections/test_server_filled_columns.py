"""a row whose L3 write left columns for the database to fill is cached as L3 holds it, or not at all.

The defect: a save naming only some of a table's columns -- the rest taken from their server
defaults -- cached the dict it wrote on every tier. L1, L2 and every replica reading L2 then served
a row without the columns the database filled in, and a reader through the Collection failed on
the missing attribute. Found in the hub's data-space ledger (``GET /admin/v1/data-spaces/{agent}``
answered 500, ``AttributeError: no attribute 'target_version'``) in the pre-PR live validation,
2026-09-30; the class lived here, in the write path every Collection shares.

What this pins:

- a synchronous save or assignment whose row the database completed is read back from L3, and
  every tier and the saving handle hold the whole row;
- a whole row is cached as written, with no read back;
- a read back that fails caches nothing, and the save still succeeds;
- a write that reaches L2 before L3 -- write-behind, a compare-and-swap -- cannot be read back,
  so a row the database would complete is refused before any tier takes it;
- a collection with no L3 has no database to fill anything, and caches the row as written.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    l2_order_columns,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "server-filled-principal"
_TABLE = "ledger_rows"
_ID = "space-1"
_KEY = f"{_SCOPE}.{_TABLE}.{_ID}"

#: what the database fills in for a column the write did not name
_SERVER_DEFAULTS: dict[str, Any] = {"target_version": 0, "max_tables": 50}


def _columns() -> list[Column]:
    return [
        Column("id", STRING_TYPE),
        Column("label", STRING_TYPE),
        Column("target_version", INT_TYPE, server_default="0"),
        Column("max_tables", INT_TYPE, server_default="50"),
        Column("note", STRING_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE, nullable=True),
    ]


_SCHEMA = TableSchema(name=_TABLE, primary_key="id", columns=_columns())
_ORDERED_SCHEMA = TableSchema(name=_TABLE, primary_key="id", columns=[*_columns(), *l2_order_columns()])


class _Row(BaseEntity):
    primary_key_field = "id"


# parity-with: threetears.core.backends.protocol.DurableStore
class _FakeDefaultingStore:
    """an in-process L3 that completes an insert as Postgres does: a server default, else NULL.

    An update (a conflicting insert, or a fenced write) changes only the columns it names, so a
    column the write left out keeps its stored value.
    """

    def __init__(self, schema: TableSchema) -> None:
        self._schema = schema
        self.rows: dict[str, dict[str, Any]] = {}
        self.fetches = 0
        self.fail_fetch = False

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        self.fetches += 1
        await asyncio.sleep(0)
        if self.fail_fetch:
            raise ConnectionError("L3 unreachable")
        row = self.rows.get(str(pk["id"]))
        return None if row is None else dict(row)

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str] | None = None,
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        await asyncio.sleep(0)
        key = str(row["id"])
        stored = self.rows.get(key)
        if cas is not None and (stored is None or stored["date_updated"] != cas):
            return 0
        if stored is None:
            whole = {
                c.name: row[c.name] if c.name in row else _SERVER_DEFAULTS.get(c.name) for c in self._schema.columns
            }
        else:
            whole = dict(stored)
            whole.update({c.name: row[c.name] for c in self._schema.mutable_columns() if c.name in row})
        self.rows[key] = whole
        return 1

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        self.rows.pop(str(pk["id"]), None)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows.values()]


class _FakeOrderedDefaultingStore(_FakeDefaultingStore):
    """the store above, able to persist a compare-and-swap row fenced on its order."""

    async def upsert_ordered(self, table: str, row: Mapping[str, Any], *, conn: Any = None) -> int:
        return await self.upsert(table, row, conn=conn)


class _Ledger(SchemaBackedCollection[_Row]):
    """a three-tier collection whose table fills two columns from server defaults."""

    primary_key_column = "id"
    schema = _SCHEMA
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "synchronous"

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _BufferedLedger(_Ledger):
    """the ledger above, writing L3 behind L2 through a write buffer."""

    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _OrderedLedger(_Ledger):
    """the ledger above, carrying the order columns a compare-and-swap persists."""

    schema = _ORDERED_SCHEMA


def _registry(nats: FakeNatsClient, store: _FakeDefaultingStore | None, schema: TableSchema) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"server_filled_{uuid.uuid4().hex[:8]}")
    l1.initialize(schema.to_sqlalchemy_table(MetaData()).metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=store, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _ledger(nats: FakeNatsClient, store: _FakeDefaultingStore | None) -> _Ledger:
    return _Ledger(_registry(nats, store, _SCHEMA), _config())


async def _l2_row(nats: FakeNatsClient) -> dict[str, Any] | None:
    raw = await (await nats.kv_bucket(name="collections")).get(key=_KEY)
    return None if raw is None else dict(json.loads(raw))


async def _settle_background() -> None:
    """wait for every task this test scheduled -- the fire-and-forget propagation of an assignment."""
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    await asyncio.gather(*pending)


class TestTheSchemaNamesTheColumnsTheDatabaseDecides:
    def test_a_row_naming_every_column_leaves_none_to_the_database(self) -> None:
        coll = _ledger(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA))
        row = {"id": _ID, "label": "a", "target_version": 1, "max_tables": 5, "note": None}
        row |= {"date_created": None, "date_updated": None}
        assert coll.columns_decided_by_store(row) == ()

    def test_a_server_default_column_the_row_leaves_out_is_decided_by_the_database(self) -> None:
        coll = _ledger(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA))
        row = {"id": _ID, "label": "a", "max_tables": 5, "note": None, "date_created": None, "date_updated": None}
        assert coll.columns_decided_by_store(row) == ("target_version",)

    def test_a_nullable_column_the_row_leaves_out_is_decided_by_the_database(self) -> None:
        coll = _ledger(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA))
        row = {"id": _ID, "label": "a", "target_version": 1, "max_tables": 5}
        row |= {"date_created": None, "date_updated": None}
        assert coll.columns_decided_by_store(row) == ("note",)

    def test_a_nullable_column_every_generated_statement_writes_null_is_completed(self) -> None:
        coll = _ledger(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA))
        row = {"id": _ID, "label": "a", "target_version": 1, "max_tables": 5, "date_created": None}
        completed = coll.complete_written_row(row)
        assert completed == {**row, "note": None, "date_updated": None}
        assert coll.columns_decided_by_store(completed) == ()

    def test_a_collection_writing_its_own_sql_completes_nothing(self) -> None:
        coll = _L2OnlyLedger(_registry(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA), _SCHEMA), _config())
        row = {"id": _ID, "label": "a", "target_version": 1, "max_tables": 5, "date_created": None}
        assert coll.complete_written_row(row) == row
        assert coll.columns_decided_by_store(row) == ("note", "date_updated")

    def test_a_collection_wrapping_the_generated_write_declares_it_and_completes(self) -> None:
        coll = _WrappingLedger(_registry(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA), _SCHEMA), _config())
        row = {"id": _ID, "label": "a", "target_version": 1, "max_tables": 5, "date_created": None}
        assert coll.complete_written_row(row) == {**row, "note": None, "date_updated": None}

    def test_a_null_in_a_not_null_server_default_column_is_decided_by_the_database(self) -> None:
        coll = _ledger(FakeNatsClient(), _FakeDefaultingStore(_SCHEMA))
        row = {"id": _ID, "label": "a", "target_version": None, "max_tables": 5, "note": None}
        row |= {"date_created": None, "date_updated": None}
        assert coll.columns_decided_by_store(row) == ("target_version",)


class TestASynchronousWriteCachesTheRowL3Holds:
    @pytest.mark.asyncio
    async def test_a_save_relying_on_server_defaults_reads_whole_on_every_tier(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        coll, peer = _ledger(nats, store), _ledger(nats, store)
        entity = coll.create({"id": _ID, "label": "first"})
        await coll.save_entity(entity)

        assert store.rows[_ID]["target_version"] == 0, "the harness no longer fills the server default"
        l1 = coll.get_row_sync(_ID)
        assert l1 is not None
        assert (l1["target_version"], l1["max_tables"], l1["note"]) == (0, 50, None), "L1 holds the row as written"
        l2 = await _l2_row(nats)
        assert l2 is not None
        assert (l2["target_version"], l2["max_tables"]) == (0, 50), "L2 holds the row as written"
        assert entity.target_version == 0, "the saving handle holds the row as written"
        served = await peer.get(_ID)
        assert served is not None
        assert (served.target_version, served.max_tables) == (0, 50), "a replica reading L2 got the partial row"

    @pytest.mark.asyncio
    async def test_a_whole_row_is_cached_without_reading_l3_back(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        coll = _ledger(nats, store)
        await coll.save_entity(
            coll.create({"id": _ID, "label": "whole", "target_version": 3, "max_tables": 7, "note": "n"})
        )
        assert store.fetches == 0, "a whole row was read back"
        l1 = coll.get_row_sync(_ID)
        assert l1 is not None and l1["target_version"] == 3

    @pytest.mark.asyncio
    async def test_a_row_leaving_out_only_known_nulls_is_completed_without_reading_l3_back(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        coll = _ledger(nats, store)
        await coll.save_entity(coll.create({"id": _ID, "label": "nulls", "target_version": 3, "max_tables": 7}))
        assert store.fetches == 0, "a row of known nulls was read back"
        l2 = await _l2_row(nats)
        assert l2 is not None and "note" in l2 and l2["note"] is None, "L2 holds the row without its nulls"

    @pytest.mark.asyncio
    async def test_an_update_leaving_out_a_column_caches_the_stored_value(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        coll = _ledger(nats, store)
        await coll.save_entity(coll.create({"id": _ID, "label": "first", "note": "kept"}))
        await coll.invalidate_cache(_ID)
        updated = coll.create({"id": _ID, "label": "second", "target_version": 4})
        await coll.save_entity(updated)

        assert store.rows[_ID]["note"] == "kept", "the harness no longer keeps a column an update left out"
        l1 = coll.get_row_sync(_ID)
        assert l1 is not None
        assert (l1["label"], l1["target_version"], l1["note"]) == ("second", 4, "kept")
        l2 = await _l2_row(nats)
        assert l2 is not None and l2["note"] == "kept"

    @pytest.mark.asyncio
    async def test_an_assignment_relying_on_server_defaults_reads_whole_on_every_tier(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        coll, peer = _ledger(nats, store), _ledger(nats, store)
        coll[_ID] = {"id": _ID, "label": "assigned"}
        await _settle_background()

        l1 = coll.get_row_sync(_ID)
        assert l1 is not None and (l1["target_version"], l1["max_tables"]) == (0, 50)
        l2 = await _l2_row(nats)
        assert l2 is not None and (l2["target_version"], l2["max_tables"]) == (0, 50)
        served = await peer.get(_ID)
        assert served is not None and served.max_tables == 50

    @pytest.mark.asyncio
    async def test_a_failed_read_back_caches_nothing_and_the_save_still_succeeds(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        coll = _ledger(nats, store)
        store.fail_fetch = True
        await coll.save_entity(coll.create({"id": _ID, "label": "unread"}))

        assert _ID in store.rows, "the save did not reach L3"
        assert coll.get_row_sync(_ID) is None, "L1 kept a row L3 completed"
        assert await _l2_row(nats) is None, "L2 took a row L3 completed"
        store.fail_fetch = False
        served = await coll.get(_ID)
        assert served is not None and served.target_version == 0


class TestAWriteAheadOfL3RefusesARowTheDatabaseWouldComplete:
    @pytest.mark.asyncio
    async def test_a_write_behind_save_relying_on_server_defaults_is_refused(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        buffer = WriteBuffer()
        coll = _BufferedLedger(_registry(nats, store, _SCHEMA), _config(), write_buffer=buffer)
        with pytest.raises(ValueError, match="target_version"):
            await coll.save_entity(coll.create({"id": _ID, "label": "buffered"}))

        assert coll.get_row_sync(_ID) is None, "L1 kept a row the database would complete"
        assert await _l2_row(nats) is None, "L2 took a row the database would complete"
        assert buffer.pending_count() == 0, "a refused row was buffered for L3"

    @pytest.mark.asyncio
    async def test_a_write_behind_assignment_relying_on_server_defaults_is_refused(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        buffer = WriteBuffer()
        coll = _BufferedLedger(_registry(nats, store, _SCHEMA), _config(), write_buffer=buffer)
        with pytest.raises(ValueError, match="target_version"):
            coll[_ID] = {"id": _ID, "label": "buffered"}
        await _settle_background()

        assert coll.get_row_sync(_ID) is None
        assert await _l2_row(nats) is None
        assert buffer.pending_count() == 0

    @pytest.mark.asyncio
    async def test_a_write_behind_save_of_a_whole_row_is_cached_as_written(self) -> None:
        nats, store = FakeNatsClient(), _FakeDefaultingStore(_SCHEMA)
        buffer = WriteBuffer()
        coll = _BufferedLedger(_registry(nats, store, _SCHEMA), _config(), write_buffer=buffer)
        await coll.save_entity(
            coll.create({"id": _ID, "label": "whole", "target_version": 2, "max_tables": 9, "note": None})
        )
        l2 = await _l2_row(nats)
        assert l2 is not None and l2["target_version"] == 2
        assert buffer.pending_count() == 1

    @pytest.mark.asyncio
    async def test_a_compare_and_swap_relying_on_server_defaults_is_refused(self) -> None:
        nats, store = FakeNatsClient(), _FakeOrderedDefaultingStore(_ORDERED_SCHEMA)
        coll = _OrderedLedger(_registry(nats, store, _ORDERED_SCHEMA), _config())
        with pytest.raises(ValueError, match="target_version"):
            await coll.l2_cas_mutate(_ID, lambda _row: ("upsert", {"id": _ID, "label": "swapped"}))

        assert await _l2_row(nats) is None, "L2 took a row the database would complete"
        assert _ID not in store.rows

    @pytest.mark.asyncio
    async def test_a_compare_and_swap_of_a_whole_row_lands(self) -> None:
        nats, store = FakeNatsClient(), _FakeOrderedDefaultingStore(_ORDERED_SCHEMA)
        coll = _OrderedLedger(_registry(nats, store, _ORDERED_SCHEMA), _config())
        whole = {"id": _ID, "label": "swapped", "target_version": 1, "max_tables": 2, "note": None}
        outcome = await coll.l2_cas_mutate(_ID, lambda _row: ("upsert", dict(whole)))
        assert outcome.action == "created"
        assert store.rows[_ID]["label"] == "swapped"


class _L2OnlyLedger(_Ledger):
    """the ledger above wired with no L3, reporting a save done as a coordination table does."""

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return 1


class _WrappingLedger(_L2OnlyLedger):
    """the override above, declared as writing through the generated statement."""

    stores_through_generated_sql: ClassVar[bool | None] = True


class TestACollectionWithoutL3CachesTheRowAsWritten:
    @pytest.mark.asyncio
    async def test_nothing_is_read_back_where_there_is_no_database(self) -> None:
        nats = FakeNatsClient()
        coll = _L2OnlyLedger(_registry(nats, None, _SCHEMA), _config())
        await coll.save_entity(coll.create({"id": _ID, "label": "l2-only"}))
        l1 = coll.get_row_sync(_ID)
        assert l1 is not None and l1["label"] == "l2-only"
        l2 = await _l2_row(nats)
        assert l2 is not None and l2["label"] == "l2-only"
