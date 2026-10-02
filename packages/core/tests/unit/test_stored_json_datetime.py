"""one stored form for a datetime inside JSON, at every tier.

A datetime nested in a JSON value used to be written three ways: L3's jsonb codec used
``default=str`` (``2026-10-01 12:30:00+00:00``), L2's payload encoders and L1's cache used
``isoformat()`` (``2026-10-01T12:30:00+00:00``), and ``isoformat()`` drops the fraction when it
is zero, so even one tier wrote two widths. The same instant read back as a different string
depending on which tier answered, and stored strings did not sort as the instants they name.

Every storage encoder now writes :func:`~threetears.core.serialization.json_datetime`'s form:
ISO 8601 extended, ``T`` separator, always microseconds, explicit UTC offset -- fixed width, so
stored strings compare and sort correctly. UUID and Decimal encode exactly as before.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from enum import StrEnum
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB

from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.schema_sql import json_default
from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.asyncpg_init import register_jsonb_text_codec
from threetears.core.serialization import json_datetime, serialize_to_json, to_stored_json
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    UUID_TYPE,
    Column as SchemaColumn,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.schema import ColumnDef, TableDef
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

_WHEN = datetime(2026, 10, 1, 12, 30, tzinfo=UTC)
_CANONICAL = "2026-10-01T12:30:00.000000+00:00"
_ID = UUID("01926f3a-0000-7000-8000-000000000001")
_NESTED: dict[str, Any] = {"answers": {"q1": {"date_answered": _WHEN, "session_id": _ID, "score": Decimal("2.50")}}}
#: what every tier stores for ``_NESTED``: the datetime in the one form, the UUID and the Decimal
#: as their strings, as they always were.
_STORED: dict[str, Any] = {"answers": {"q1": {"date_answered": _CANONICAL, "session_id": str(_ID), "score": "2.50"}}}


class TestTheOneForm:
    def test_aware_utc_is_written_with_t_microseconds_and_offset(self) -> None:
        assert json_datetime(_WHEN) == _CANONICAL

    def test_a_fraction_keeps_all_six_digits(self) -> None:
        assert json_datetime(_WHEN.replace(microsecond=1200)) == "2026-10-01T12:30:00.001200+00:00"

    def test_an_aware_non_utc_value_is_converted_to_utc_first(self) -> None:
        eastern = datetime(2026, 10, 1, 8, 30, tzinfo=timezone(timedelta(hours=-4)))
        assert json_datetime(eastern) == _CANONICAL

    def test_the_form_is_fixed_width_so_strings_sort_as_instants(self) -> None:
        instants = [_WHEN + timedelta(microseconds=step) for step in (0, 1, 999_999, 1_000_000)]
        written = [json_datetime(when) for when in instants]
        assert len({len(text) for text in written}) == 1
        assert sorted(written) == written

    def test_it_parses_back_to_the_same_instant(self) -> None:
        assert datetime.fromisoformat(json_datetime(_WHEN)) == _WHEN

    def test_every_legacy_stored_form_still_parses_to_the_same_instant(self) -> None:
        """rows written before this keep their strings; the readers' parser accepts every one."""
        for legacy in (str(_WHEN), _WHEN.isoformat(), "2026-10-01T12:30:00Z"):
            assert datetime.fromisoformat(legacy) == _WHEN

    def test_a_naive_value_is_refused_and_the_error_says_what_to_do(self) -> None:
        """a naive datetime names no instant: storing it would make every reader guess the zone."""
        with pytest.raises(ValueError, match=r"naive datetime .*names no instant.*timezone-aware"):
            json_datetime(datetime(2026, 10, 1, 12, 30))

    def test_the_refusal_names_the_field_when_the_caller_knows_it(self) -> None:
        with pytest.raises(ValueError, match=r"naive datetime in 'answers\.q1\.date_answered'"):
            json_datetime(datetime(2026, 10, 1, 12, 30), field="answers.q1.date_answered")

    def test_a_tzinfo_that_answers_no_offset_is_refused_as_naive(self) -> None:
        """``tzinfo`` set but ``utcoffset()`` ``None`` is naive by Python's own definition."""

        class NoOffset(tzinfo):
            def utcoffset(self, dt: datetime | None) -> timedelta | None:
                return None

            def dst(self, dt: datetime | None) -> timedelta | None:
                return None

        with pytest.raises(ValueError, match="naive datetime"):
            json_datetime(datetime(2026, 10, 1, 12, 30, tzinfo=NoOffset()))


class TestTheSharedHandler:
    def test_a_naive_datetime_nested_anywhere_is_refused_at_every_tier(self) -> None:
        naive = {"answers": {"q1": {"date_answered": datetime(2026, 10, 1, 12, 30)}}}
        with pytest.raises(ValueError, match="naive datetime"):
            json.dumps(naive, default=json_default)
        with pytest.raises(ValueError, match="naive datetime"):
            serialize_to_json(naive)
        with pytest.raises(ValueError, match="naive datetime"):
            _l3_encoders()["jsonb"](naive)

    def test_a_date_is_its_iso_string_as_default_str_wrote_it(self) -> None:
        assert json.dumps({"d": date(2026, 10, 1)}, default=json_default) == json.dumps({"d": str(date(2026, 10, 1))})

    def test_uuid_and_decimal_are_unchanged(self) -> None:
        assert json.loads(json.dumps({"u": _ID, "n": Decimal("19.99")}, default=json_default)) == {
            "u": str(_ID),
            "n": "19.99",
        }


def _l3_encoders() -> dict[str, Any]:
    """the encoders L3's codec registers, read off a connection that records them.

    :return: the encoder registered per type name
    :rtype: dict[str, Any]
    """
    encoders: dict[str, Any] = {}

    class RecordingConnection:
        async def set_type_codec(self, typename: str, **kwargs: Any) -> None:
            encoders[typename] = kwargs["encoder"]

    asyncio.run(register_jsonb_text_codec(RecordingConnection()))
    return encoders


def _sqlite_with_a_json_column() -> SQLiteBackend:
    """an in-memory L1 holding one table with a JSON column.

    :return: the initialized backend
    :rtype: SQLiteBackend
    """
    metadata = MetaData()
    Table("rows", metadata, Column("id", String, primary_key=True), Column("data", JSONB))
    backend = SQLiteBackend(db_name="stored_json_datetime")
    backend.initialize(metadata)
    return backend


class TestEveryTierStoresTheSameString:
    """the same nested value, through each tier's storage encoder, stores and reads back the same text."""

    def test_l3_jsonb_and_json_codecs(self) -> None:
        encoders = _l3_encoders()
        for typename in ("jsonb", "json"):
            assert json.loads(encoders[typename](_NESTED)) == _STORED

    def test_l3_through_the_broker(self) -> None:
        """an agent's jsonb write reaches the broker already encoded; the broker stores what arrives."""
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps({"success": True, "rowcount": 1}).encode("utf-8"))
        backend = NatsProxyL3Backend(
            nats_client=nc, namespace_prefix="test", agent_id="agent-1", identity_token=lambda: "t"
        )

        asyncio.run(backend.execute("UPDATE t SET doc = $1", _NESTED))

        sent = json.loads(nc.request_raw.call_args.kwargs["payload"])
        assert sent["params"] == [_STORED]

    def test_l2_payload_encoder(self) -> None:
        assert json.loads(json.dumps(_NESTED, default=json_default)) == _STORED

    def test_l2_entity_codec(self) -> None:
        assert json.loads(serialize_to_json(_NESTED)) == _STORED

    def test_l1_sqlite_round_trip(self) -> None:
        backend = _sqlite_with_a_json_column()
        try:
            backend.upsert("rows", {"id": "r1", "data": _NESTED})
            row = backend.select_by_id("rows", "r1")
        finally:
            backend.reset()
        assert row is not None
        assert row["data"] == _STORED

    def test_l1_duckdb(self) -> None:
        assert json.loads(DuckDBBackend().serialize_value(_NESTED, "VARCHAR_JSON")) == _STORED

    @pytest.mark.parametrize("value", [_NESTED, [_NESTED]])
    def test_all_tiers_agree_byte_for_byte(self, value: Any) -> None:
        l3 = _l3_encoders()["jsonb"](value)
        l2 = json.dumps(value, default=json_default)
        l1 = SQLiteBackend().serialize_value(value, "TEXT_JSON")
        assert l3 == l2 == l1


class _Colour(StrEnum):
    RED = "red"


class _Inner(BaseModel):
    date_seen: datetime
    colour: _Colour


class _Outer(BaseModel):
    run_id: UUID
    cost: Decimal
    tags: frozenset[str] = Field(default_factory=frozenset)
    inner: list[_Inner]
    date_finished: datetime | None = None


class TestStoredJsonOfAModel:
    """``to_stored_json`` is ``model_dump(mode="json")`` with every datetime in the one stored form.

    Pydantic's JSON mode writes ``2026-10-01T12:30:00Z``, dropping a zero fraction, so a model dumped
    for a JSONB column or a KV value stored a second spelling of the same instant.
    """

    def _outer(self) -> _Outer:
        return _Outer(
            run_id=_ID,
            cost=Decimal("2.50"),
            tags=frozenset({"a"}),
            inner=[_Inner(date_seen=_WHEN, colour=_Colour.RED)],
            date_finished=_WHEN.replace(microsecond=1200),
        )

    def test_datetimes_take_the_one_form_at_every_depth(self) -> None:
        stored = to_stored_json(self._outer())
        assert stored["inner"][0]["date_seen"] == _CANONICAL
        assert stored["date_finished"] == "2026-10-01T12:30:00.001200+00:00"

    def test_everything_else_is_exactly_what_json_mode_writes(self) -> None:
        outer = self._outer()
        stored = to_stored_json(outer)
        json_mode = outer.model_dump(mode="json")
        stored["inner"][0].pop("date_seen")
        json_mode["inner"][0].pop("date_seen")
        stored.pop("date_finished")
        json_mode.pop("date_finished")
        assert stored == json_mode

    def test_pydantic_json_mode_alone_writes_the_other_spelling(self) -> None:
        """the gap this closes: the same instant, stored by json mode, is a different string."""
        assert self._outer().model_dump(mode="json")["inner"][0]["date_seen"] != _CANONICAL

    def test_a_plain_structure_holding_models_and_datetimes(self) -> None:
        stored = to_stored_json({"when": _WHEN, "items": [_Inner(date_seen=_WHEN, colour=_Colour.RED)], "id": _ID})
        assert stored == {"when": _CANONICAL, "items": [{"date_seen": _CANONICAL, "colour": "red"}], "id": str(_ID)}

    def test_the_result_round_trips_through_json(self) -> None:
        stored = to_stored_json(self._outer())
        assert json.loads(json.dumps(stored)) == stored
        assert _Outer.model_validate(stored) == self._outer()

    def test_a_naive_datetime_is_refused_naming_where_it_sits(self) -> None:
        naive = {"runs": [{"date_started": datetime(2026, 10, 1, 12, 30)}]}
        with pytest.raises(ValueError, match=r"naive datetime in 'runs\[0\]\.date_started'"):
            to_stored_json(naive)

    def test_legacy_spellings_still_validate_into_the_model(self) -> None:
        """rows stored before this keep their ``...Z`` strings; reading them is unchanged."""
        legacy = self._outer().model_dump(mode="json")
        assert _Outer.model_validate(legacy) == self._outer()


class _TzEntity(BaseEntity):
    primary_key_field = "id"


class _TzRows(SchemaBackedCollection[_TzEntity]):
    primary_key_column: str = "id"
    schema = TableSchema(
        name="tz_rows",
        primary_key="id",
        columns=[
            SchemaColumn("id", UUID_TYPE),
            SchemaColumn("doc", JSONB_TYPE, nullable=True),
            SchemaColumn("date_seen", DATETIMETZ_TYPE, nullable=True),
            SchemaColumn("date_created", DATETIMETZ_TYPE),
            SchemaColumn("date_updated", DATETIMETZ_TYPE),
        ],
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "tz_rows"

    @property
    def entity_class(self) -> type[_TzEntity]:
        """return entity class."""
        return _TzEntity


class TestADeclaredTimestampColumnIsNotARefusal:
    """a collection's own timestamp columns reach the encoder normalised, so the refusal is for documents."""

    def test_a_schema_backed_collection_declares_its_instant_columns_from_its_schema(self) -> None:
        """the schema already names them; restating them as ``datetime_columns`` is how one gets left out."""
        assert _TzRows.datetime_columns == frozenset({"date_seen", "date_created", "date_updated"})

    @pytest.mark.asyncio
    async def test_a_naive_column_value_reaches_l2_as_the_instant_l3_stores(self) -> None:
        """L3's write coercion reads a naive TIMESTAMPTZ value as UTC; the L2 payload now agrees."""
        nats = FakeNatsClient()
        pool = MagicMock()
        pool.execute = AsyncMock(return_value="INSERT 0 1")
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool, l2_client=nats, kv_key_scope="stored-json-datetime")
        collection = _TzRows(
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        )
        naive = datetime(2026, 10, 1, 12, 30)
        entity = collection.create({"id": _ID, "doc": {"k": 1}, "date_seen": naive, "date_updated": _WHEN})

        await collection.save_entity(entity)

        bucket = await nats.kv_bucket(name=collection.L2_BUCKET_SUFFIX)
        stored = await bucket.get(key=collection.l2_key(_ID))
        assert stored is not None
        assert json.loads(stored)["date_seen"] == _CANONICAL

    def test_a_dynamic_tables_wall_clock_column_stays_naive_through_l2(self) -> None:
        """a product table's ``timestamp`` column is naive by declaration: cached as its wall-clock text."""
        table = TableDef(
            name="readings",
            columns=[
                ColumnDef(name="id", column_type="text", primary_key=True),
                ColumnDef(name="taken", column_type="timestamp"),
                ColumnDef(name="recorded", column_type="timestamptz"),
            ],
        )
        collection = create_dynamic_collection(
            table_def=table,
            registry=CollectionRegistry(),
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
            nats_client=None,
        )
        row = {"id": "r1", "taken": datetime(2026, 10, 1, 12, 30), "recorded": _WHEN}

        encoded = collection.serialize(row)

        assert json.loads(encoded) == {"id": "r1", "taken": "2026-10-01T12:30:00.000000", "recorded": _CANONICAL}
        assert collection.deserialize(encoded) == row
        assert type(collection).datetime_columns == frozenset({"recorded"})

    def test_a_dynamic_tables_instant_column_refuses_a_naive_value_outside_the_write_path(self) -> None:
        table = TableDef(
            name="readings2",
            columns=[
                ColumnDef(name="id", column_type="text", primary_key=True),
                ColumnDef(name="recorded", column_type="timestamptz"),
            ],
        )
        collection = create_dynamic_collection(
            table_def=table,
            registry=CollectionRegistry(),
            config=DefaultCoreConfig(collection_flush="ALWAYS"),
            nats_client=None,
        )
        with pytest.raises(ValueError, match="naive datetime"):
            collection.serialize({"id": "r1", "recorded": datetime(2026, 10, 1, 12, 30)})
