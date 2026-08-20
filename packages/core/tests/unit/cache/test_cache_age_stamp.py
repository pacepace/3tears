"""The injected L1 cache-age stamp column.

Chunk 02 lands the column and writes it on pull-through. Nothing reads it for
expiry yet -- these tests pin that it exists where it must, is written when it
should be, and is invisible to every caller above the cache tier.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table

from threetears.core.cache.base import _CACHED_AT_COLUMN
from threetears.core.cache.sqlite import SQLiteBackend


def _metadata(table: str = "widgets", *, with_reserved_column: bool = False) -> MetaData:
    metadata = MetaData()
    columns = [
        Column("id", String(64), primary_key=True),
        Column("name", String(255)),
        Column("size", Integer),
    ]
    if with_reserved_column:
        columns.append(Column(_CACHED_AT_COLUMN, Integer))
    Table(table, metadata, *columns)
    return metadata


def _backend(metadata: MetaData) -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"stamp_{uuid.uuid4().hex[:8]}")
    b.initialize(metadata)
    return b


def _declared_columns(backend: SQLiteBackend, table: str) -> set[str]:
    """Column names SQLite itself reports for ``table``."""
    conn = backend.get_connection()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


class TestTheColumnLandsInBothPlaces:
    """The DDL alone is not enough, and that is the trap this guards.

    ``upsert`` filters writes to the schema registry, so a column present only
    in the table would silently discard every stamp write and stay NULL
    forever -- with a DDL-only test passing throughout.
    """

    def test_the_table_has_the_column(self) -> None:
        b = _backend(_metadata())
        assert _CACHED_AT_COLUMN in _declared_columns(b, "widgets")

    def test_the_registry_has_it_too_or_this_write_would_vanish(self) -> None:
        """The registry half, asserted through behaviour rather than by reaching in.

        ``upsert`` filters writes to the schema registry, so a stamp that
        survives a write-then-read round trip proves the registry knows the
        column. A structural check on the registry dict would assert the same
        fact while proving less.
        """
        b = _backend(_metadata())
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 1234.5}, "id")
        conn = b.get_connection()
        stored = conn.execute(f'SELECT "{_CACHED_AT_COLUMN}" FROM widgets WHERE id = ?', ("w1",)).fetchone()
        assert stored[0] == 1234.5


class TestTheStampIsInvisibleToCallers:
    def test_select_by_id_does_not_return_it(self) -> None:
        b = _backend(_metadata())
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 1234.5}, "id")
        row = b.select_by_id("widgets", "w1", "id")
        assert row is not None
        assert _CACHED_AT_COLUMN not in row
        assert row["name"] == "one"

    def test_select_batch_does_not_return_it(self) -> None:
        b = _backend(_metadata())
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 1.0}, "id")
        b.upsert("widgets", {"id": "w2", "name": "two", _CACHED_AT_COLUMN: 2.0}, "id")
        rows = b.select_batch("widgets", ["w1", "w2"], "id")
        assert len(rows) == 2
        assert all(_CACHED_AT_COLUMN not in r for r in rows)

    def test_a_caller_naming_it_explicitly_still_does_not_get_it(self) -> None:
        """No projection reaches past the strip.

        The column is in the schema registry, so ``build_select_clause``
        accepts the name rather than raising -- which is exactly why the strip
        has to be at the deserialize funnel and not at the projection.
        """
        b = _backend(_metadata())
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 1234.5}, "id")
        row = b.select_by_id("widgets", "w1", "id", columns=["name", _CACHED_AT_COLUMN])
        assert row is not None
        assert _CACHED_AT_COLUMN not in row
        assert row["name"] == "one"


class TestLocallyAuthoredRows:
    def test_a_write_with_no_stamp_leaves_it_null(self) -> None:
        """A row this process authored has no provenance from a lower tier."""
        b = _backend(_metadata())
        b.upsert("widgets", {"id": "w1", "name": "one"}, "id")
        conn = b.get_connection()
        stored = conn.execute(f'SELECT "{_CACHED_AT_COLUMN}" FROM widgets WHERE id = ?', ("w1",)).fetchone()
        assert stored[0] is None

    def test_read_modify_write_does_not_clear_an_existing_stamp(self) -> None:
        """The read-modify-write pattern must not silently reset provenance.

        ``agent/tools/collections.py`` reads a row from L1, edits one field and
        upserts it straight back. Reads strip the stamp, so that write-back
        carries none. If absence cleared the column, an hours-old row would
        start reading as locally-authored -- and locally-authored rows never
        expire, which would quietly disable the whole mechanism for exactly
        the rows most likely to be touched.
        """
        b = _backend(_metadata())
        b.upsert("widgets", {"id": "w1", "name": "one", _CACHED_AT_COLUMN: 1234.5}, "id")

        round_tripped = b.select_by_id("widgets", "w1", "id")
        assert round_tripped is not None
        round_tripped["name"] = "edited"
        b.upsert("widgets", round_tripped, "id")

        conn = b.get_connection()
        row = conn.execute(f'SELECT name, "{_CACHED_AT_COLUMN}" FROM widgets WHERE id = ?', ("w1",)).fetchone()
        assert row[0] == "edited"
        assert row[1] == 1234.5


class TestExemptTables:
    """Tables that ride the same backend but are not entity caches."""

    @pytest.mark.parametrize("table", ["collection_scan_cache", "write_buffer"])
    def test_no_stamp_column_is_injected(self, table: str) -> None:
        b = _backend(_metadata(table))
        assert _CACHED_AT_COLUMN not in _declared_columns(b, table)

    @pytest.mark.parametrize("table", ["collection_scan_cache", "write_buffer"])
    def test_a_stray_stamp_key_is_filtered_rather_than_erroring(self, table: str) -> None:
        """The registry half of the exemption, asserted through behaviour.

        The column does not exist on these tables, so if the write were not
        filtered to the registry SQLite would raise ``no such column``. Not
        raising is the proof the registry excludes it.
        """
        b = _backend(_metadata(table))
        b.upsert(table, {"id": "x1", "name": "one", _CACHED_AT_COLUMN: 1.0}, "id")
        row = b.select_by_id(table, "x1", "id")
        assert row is not None
        assert row["name"] == "one"


class TestTheNameIsReserved:
    def test_a_table_declaring_the_column_is_rejected(self) -> None:
        """A collision is an error, not a silent duplicate-column DDL failure."""
        with pytest.raises(ValueError, match=_CACHED_AT_COLUMN):
            _backend(_metadata(with_reserved_column=True))
