"""L1 max-age expiry at the backend tier.

Every case injects both clock readings. Nothing sleeps: a test that slept to
prove an hour-long window would take an hour, and one that slept to prove a
short window would be flaky.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table

from threetears.core.cache.base import _CACHED_AT_COLUMN
from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.cache.sqlite import SQLiteBackend


def _metadata(table: str = "widgets") -> MetaData:
    metadata = MetaData()
    Table(
        table,
        metadata,
        Column("id", String(64), primary_key=True),
        Column("name", String(255)),
        Column("size", Integer),
    )
    return metadata


def _backend(table: str = "widgets") -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"expiry_{uuid.uuid4().hex[:8]}")
    b.initialize(_metadata(table))
    return b


def _row_count(backend: SQLiteBackend, table: str = "widgets") -> int:
    conn = backend.get_connection()
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


class TestExpiryIsOffUnlessAsked:
    def test_no_bound_serves_an_ancient_row(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 0.0}, "id")
        assert b.select_by_id("widgets", "w1", "id", now_monotonic=1_000_000.0) is not None

    def test_no_bound_deletes_nothing(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 0.0}, "id")
        b.select_by_id("widgets", "w1", "id", now_monotonic=1_000_000.0)
        assert _row_count(b) == 1


class TestExpiryOnRead:
    def test_a_row_inside_the_window_is_served(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 100.0}, "id")
        row = b.select_by_id("widgets", "w1", "id", max_age_seconds=30.0, now_monotonic=129.0)
        assert row is not None
        assert row["name"] == "one"

    def test_a_row_past_the_window_reads_as_a_miss(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 100.0}, "id")
        assert b.select_by_id("widgets", "w1", "id", max_age_seconds=30.0, now_monotonic=131.0) is None

    def test_an_expired_row_is_deleted_not_left_as_a_tombstone(self) -> None:
        """Otherwise every later read re-evaluates a row that can never win.

        Matters most when the row is genuinely gone from L3: the pull-through
        returns nothing, so nothing overwrites the entry, and it would sit
        there being re-judged forever.
        """
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 100.0}, "id")
        b.select_by_id("widgets", "w1", "id", max_age_seconds=30.0, now_monotonic=131.0)
        assert _row_count(b) == 0

    def test_an_hour_long_window_is_exercised_without_waiting_an_hour(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 0.0}, "id")
        assert b.select_by_id("widgets", "w1", "id", max_age_seconds=3600.0, now_monotonic=3599.0) is not None
        assert b.select_by_id("widgets", "w1", "id", max_age_seconds=3600.0, now_monotonic=3601.0) is None


class TestLocallyAuthoredRowsNeverExpire:
    def test_an_unstamped_row_survives_any_window(self) -> None:
        """It holds a write this pod made that no lower tier has served.

        Expiring it would replace a local write with the older value a
        pull-through returns, which is data loss dressed as a cache miss.
        """
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "local"}, "id")
        row = b.select_by_id("widgets", "w1", "id", max_age_seconds=0.001, now_monotonic=1_000_000.0)
        assert row is not None
        assert row["name"] == "local"
        assert _row_count(b) == 1


class TestProjectionCannotBypassExpiry:
    def test_a_projected_read_still_expires(self) -> None:
        """A caller naming columns must not silently opt out of the bound.

        The stamp is absent from a projection that does not name it, and an
        absent stamp reads as fresh, so the projection is widened internally
        rather than trusted.
        """
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 100.0}, "id")
        row = b.select_by_id("widgets", "w1", "id", columns=["name"], max_age_seconds=30.0, now_monotonic=131.0)
        assert row is None

    def test_the_widened_projection_is_still_stripped(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 100.0}, "id")
        row = b.select_by_id("widgets", "w1", "id", columns=["name"], max_age_seconds=30.0, now_monotonic=101.0)
        assert row == {"name": "one"}


class TestSelectBatchAppliesTheSamePredicate:
    def test_expired_rows_are_omitted_and_fresh_ones_kept(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "old", _CACHED_AT_COLUMN: 0.0}, "id")
        b.upsert("widgets", {"id": "w2", "name": "new", _CACHED_AT_COLUMN: 100.0}, "id")
        rows = b.select_batch("widgets", ["w1", "w2"], "id", max_age_seconds=30.0, now_monotonic=110.0)
        assert [r["name"] for r in rows] == ["new"]

    def test_expired_rows_are_deleted_by_the_batch_path_too(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "old", _CACHED_AT_COLUMN: 0.0}, "id")
        b.upsert("widgets", {"id": "w2", "name": "new", _CACHED_AT_COLUMN: 100.0}, "id")
        b.select_batch("widgets", ["w1", "w2"], "id", max_age_seconds=30.0, now_monotonic=110.0)
        assert _row_count(b) == 1

    def test_no_bound_keeps_everything(self) -> None:
        b = _backend()
        b.upsert("widgets", {"id": "w1", "name": "old", _CACHED_AT_COLUMN: 0.0}, "id")
        b.upsert("widgets", {"id": "w2", "name": "new", _CACHED_AT_COLUMN: 100.0}, "id")
        rows = b.select_batch("widgets", ["w1", "w2"], "id", now_monotonic=1_000_000.0)
        assert len(rows) == 2


class TestExemptTablesAreNeverExpired:
    """They carry no stamp, so an age question about them is meaningless."""

    @pytest.mark.parametrize("table", ["collection_scan_cache", "write_buffer"])
    def test_a_bound_does_not_expire_an_exempt_table(self, table: str) -> None:
        b = _backend(table)
        b.upsert(table, {"id": "x1", "name": "one"}, "id")
        row = b.select_by_id(table, "x1", "id", max_age_seconds=0.001, now_monotonic=1_000_000.0)
        assert row is not None
        assert _row_count(b, table) == 1


class TestDuckDBRefusesRatherThanSilentlyNotExpiring:
    """It injects no stamp, so a bound here would be a promise it cannot keep.

    Failing loudly is the point: a backend that accepted the argument and
    never expired anything would hand back exactly the unbounded staleness the
    caller asked to be rid of, with nothing to show it had not happened.
    """

    @staticmethod
    def _duck() -> DuckDBBackend:
        b = DuckDBBackend()
        b.initialize(_metadata())
        return b

    def test_select_by_id_raises_when_a_bound_is_requested(self) -> None:
        with pytest.raises(NotImplementedError, match="max-age expiry"):
            self._duck().select_by_id("widgets", "w1", "id", max_age_seconds=30.0)

    def test_select_batch_raises_when_a_bound_is_requested(self) -> None:
        with pytest.raises(NotImplementedError, match="max-age expiry"):
            self._duck().select_batch("widgets", ["w1"], "id", max_age_seconds=30.0)

    def test_it_stays_usable_without_a_bound(self) -> None:
        b = self._duck()
        b.upsert("widgets", {"id": "w1", "name": "one"}, "id")
        assert b.select_by_id("widgets", "w1", "id") is not None
