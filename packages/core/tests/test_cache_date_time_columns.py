"""Date and Time columns round-trip exactly through the SQLite and DuckDB cache backends.

A ``sqlalchemy.Time`` column (metallm's ``users.config_sleep_from``) used to fall through both type
mappers to plain text: every start logged "Unknown SQLAlchemy type ... Time", and a ``time`` came back
as a string. These run the real backends, as the other mapper tests do.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from datetime import date, time, timedelta, timezone
from typing import Any
from unittest import mock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Column, Date, Interval, MetaData, Table, Time
from sqlalchemy.dialects.postgresql import UUID

from threetears.core.cache import sqlite as sqlite_module
from threetears.core.cache.base import stored_time
from threetears.core.cache.sqlite import SQLiteBackend


def _sqlite() -> Any:
    return SQLiteBackend(db_name=f"test_date_time_{uuid.uuid4().hex[:8]}")


def _duckdb() -> Any:
    pytest.importorskip("duckdb")
    from threetears.core.cache.duckdb import DuckDBBackend

    return DuckDBBackend()


def _metadata(*extra: Column) -> MetaData:
    metadata = MetaData()
    Table(
        "clock_rows",
        metadata,
        Column("id", UUID, primary_key=True),
        Column("day", Date),
        Column("wall", Time),
        Column("zoned", Time(timezone=True)),
        *extra,
    )
    return metadata


@pytest.fixture(params=[_sqlite, _duckdb], ids=["sqlite", "duckdb"])
def make_backend(request: pytest.FixtureRequest) -> Iterator[Callable[[MetaData], Any]]:
    made: list[Any] = []

    def make(metadata: MetaData) -> Any:
        backend = request.param()
        backend.initialize(metadata)
        made.append(backend)
        return backend

    yield make
    for backend in made:
        backend.reset()


def _round_trip(backend: Any, **values: Any) -> dict[str, Any]:
    entity_id = str(uuid.uuid4())
    backend.upsert("clock_rows", {"id": entity_id, **values})
    row = backend.select_by_id("clock_rows", entity_id)
    assert row is not None
    return row


_PLUS_ONE = timezone(timedelta(hours=1))
_MINUS_FIVE_THIRTY = timezone(-timedelta(hours=5, minutes=30))


class TestRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [date(2026, 10, 28), date(1, 1, 1), date(9999, 12, 31)],
        ids=["today", "min", "max"],
    )
    def test_date(self, make_backend: Callable[[MetaData], Any], value: date) -> None:
        row = _round_trip(make_backend(_metadata()), day=value)
        assert type(row["day"]) is date
        assert row["day"] == value

    @pytest.mark.parametrize(
        "value",
        [time(22, 30), time(7, 0, 5), time(23, 59, 59, 999999), time(0, 0, 0, 1)],
        ids=["minutes", "seconds", "micro-max", "micro-min"],
    )
    def test_naive_time(self, make_backend: Callable[[MetaData], Any], value: time) -> None:
        row = _round_trip(make_backend(_metadata()), wall=value)
        assert type(row["wall"]) is time
        assert row["wall"] == value
        assert row["wall"].tzinfo is None

    def test_aware_time_in_a_time_column_keeps_the_wall_clock(self, make_backend: Callable[[MetaData], Any]) -> None:
        """Postgres ``time`` keeps the wall clock and answers it naive (asyncpg writes the fields)."""
        row = _round_trip(make_backend(_metadata()), wall=time(22, 30, 1, 5, tzinfo=_PLUS_ONE))
        assert row["wall"] == time(22, 30, 1, 5)
        assert row["wall"].tzinfo is None

    @pytest.mark.parametrize("zone", [_PLUS_ONE, _MINUS_FIVE_THIRTY, timezone.utc], ids=["+1", "-5:30", "utc"])
    def test_zoned_time_keeps_its_offset(self, make_backend: Callable[[MetaData], Any], zone: timezone) -> None:
        value = time(6, 15, 0, 250000, tzinfo=zone)
        row = _round_trip(make_backend(_metadata()), zoned=value)
        assert row["zoned"] == value
        assert row["zoned"].utcoffset() == zone.utcoffset(None)

    @pytest.mark.parametrize(
        "value",
        [time(6, 15), time(6, 15, tzinfo=ZoneInfo("Europe/Dublin"))],
        ids=["naive", "zoneinfo"],
    )
    def test_zoned_time_without_a_fixed_offset_is_refused(
        self, make_backend: Callable[[MetaData], Any], value: time
    ) -> None:
        """asyncpg refuses these for ``timetz`` before L3 writes them; the cache refuses them too."""
        backend = make_backend(_metadata())
        with pytest.raises(ValueError, match="no fixed UTC offset"):
            backend.upsert("clock_rows", {"id": str(uuid.uuid4()), "zoned": value})

    def test_stored_forms_are_iso_text(self, make_backend: Callable[[MetaData], Any]) -> None:
        """the backend writes the text itself, never leaving a date or time to a driver's adapter."""
        backend = make_backend(_metadata())
        codes = backend.column_types("clock_rows")
        assert backend.serialize_value(date(2026, 10, 28), codes["day"]) == "2026-10-28"
        assert backend.serialize_value(time(22, 30, 0, 7), codes["wall"]) == "22:30:00.000007"
        assert backend.serialize_value(time(6, 15, tzinfo=_PLUS_ONE), codes["zoned"]) == "06:15:00+01:00"

    def test_null_stays_null(self, make_backend: Callable[[MetaData], Any]) -> None:
        row = _round_trip(make_backend(_metadata()), day=None, wall=None, zoned=None)
        assert row["day"] is None
        assert row["wall"] is None
        assert row["zoned"] is None


class TestStoredTime:
    def test_text_forms(self) -> None:
        assert stored_time(time(22, 30), zoned=False) == "22:30:00"
        assert stored_time(time(22, 30, 0, 7), zoned=False) == "22:30:00.000007"
        assert stored_time(time(22, 30, tzinfo=_PLUS_ONE), zoned=False) == "22:30:00"
        assert stored_time(time(22, 30, tzinfo=_PLUS_ONE), zoned=True) == "22:30:00+01:00"


class TestUnknownTypeWarning:
    def _warnings(self, backend: Any, metadata: MetaData) -> list[str]:
        module = sqlite_module if isinstance(backend, SQLiteBackend) else _duckdb_module()
        with mock.patch.object(module.log, "warning") as warning:
            backend.initialize(metadata)
        return [str(c.args[0]) for c in warning.call_args_list]

    @pytest.mark.parametrize("factory", [_sqlite, _duckdb], ids=["sqlite", "duckdb"])
    def test_date_and_time_do_not_warn(self, factory: Callable[[], Any]) -> None:
        backend = factory()
        try:
            assert self._warnings(backend, _metadata()) == []
        finally:
            backend.reset()

    @pytest.mark.parametrize("factory", [_sqlite, _duckdb], ids=["sqlite", "duckdb"])
    def test_a_genuinely_unknown_type_still_warns(self, factory: Callable[[], Any]) -> None:
        backend = factory()
        try:
            warned = self._warnings(backend, _metadata(Column("span", Interval)))
            # the mapper runs once per use of the column (DDL, then the schema registry)
            assert warned
            assert all("Interval" in w for w in warned)
        finally:
            backend.reset()


def _duckdb_module() -> Any:
    from threetears.core.cache import duckdb as duckdb_module

    return duckdb_module
