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
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB

from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.schema_sql import json_default
from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.asyncpg_init import register_jsonb_text_codec
from threetears.core.serialization import json_datetime, serialize_to_json

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

    def test_a_naive_value_is_written_fixed_width_without_an_offset(self) -> None:
        """tolerated, not endorsed: production writers still produce one (see the docstring)."""
        assert json_datetime(datetime(2026, 10, 1, 12, 30)) == "2026-10-01T12:30:00.000000"


class TestTheSharedHandler:
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
