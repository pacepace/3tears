"""Enforcement test -- L1 cache write round-trip for all column types.

Verifies that the L1 SQLite backend (and optionally DuckDB) can create
tables from SQLAlchemy metadata, write sample data for every column type,
and read it back. Catches DDL generation bugs and serialization issues.

Does NOT require database connections -- uses in-memory backends only.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID

from threetears.core.cache.sqlite import SQLiteBackend


def _make_comprehensive_metadata() -> MetaData:
    """Create SQLAlchemy metadata covering all supported column types."""
    metadata = MetaData()
    Table(
        "roundtrip_test",
        metadata,
        Column("id", UUID, primary_key=True),
        Column("name", String(255)),
        Column("description", Text),
        Column("count", Integer),
        Column("score", Float),
        Column("active", Boolean),
        Column("data", JSONB),
        Column("created_at", DateTime(timezone=True)),
        Column("raw_bytes", BYTEA),
    )
    return metadata


def _sample_row(entity_id: str | None = None) -> dict[str, Any]:
    """Create a sample row with representative values for every column type."""
    return {
        "id": entity_id or str(uuid.uuid4()),
        "name": "Test Entity",
        "description": "A longer text description for testing",
        "count": 42,
        "score": 3.14,
        "active": True,
        "data": {"key": "value", "nested": {"a": 1}},
        "created_at": datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        "raw_bytes": b"\xde\xad\xbe\xef",
    }


@pytest.fixture()
def sqlite_backend() -> SQLiteBackend:
    """Create and initialize a SQLiteBackend, reset after test."""
    b = SQLiteBackend(db_name=f"enforcement_roundtrip_{uuid.uuid4().hex[:8]}")
    b.initialize(_make_comprehensive_metadata())
    yield b
    b.reset()


class TestSQLiteRoundTrip:
    """Write sample data to SQLite L1 and read it back for every column type."""

    def test_write_and_readback(self, sqlite_backend: SQLiteBackend) -> None:
        """Write sample data and verify readback matches."""
        row = _sample_row()
        sqlite_backend.upsert("roundtrip_test", row)

        result = sqlite_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None, "Row not found after upsert"

        # Verify each column type round-trips correctly
        assert result["name"] == row["name"]
        assert result["description"] == row["description"]
        assert result["count"] == row["count"]
        assert result["active"] is True
        assert result["data"] == row["data"]
        assert result["created_at"] == row["created_at"]
        assert result["raw_bytes"] == row["raw_bytes"]

    def test_uuid_column(self, sqlite_backend: SQLiteBackend) -> None:
        """UUID columns round-trip through serialization."""
        row = _sample_row()
        sqlite_backend.upsert("roundtrip_test", row)
        result = sqlite_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None
        assert result["id"] == uuid.UUID(row["id"])

    def test_bool_false_round_trip(self, sqlite_backend: SQLiteBackend) -> None:
        """Boolean False must not be confused with NULL or 0."""
        row = _sample_row()
        row["active"] = False
        sqlite_backend.upsert("roundtrip_test", row)
        result = sqlite_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None
        assert result["active"] is False

    def test_null_values(self, sqlite_backend: SQLiteBackend) -> None:
        """NULL values for all nullable columns."""
        row = _sample_row()
        row["name"] = None
        row["description"] = None
        row["count"] = None
        row["score"] = None
        row["active"] = None
        row["data"] = None
        row["created_at"] = None
        row["raw_bytes"] = None
        sqlite_backend.upsert("roundtrip_test", row)
        result = sqlite_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None
        assert result["name"] is None
        assert result["data"] is None
        assert result["created_at"] is None
        assert result["raw_bytes"] is None

    def test_json_complex_structure(self, sqlite_backend: SQLiteBackend) -> None:
        """Complex JSON structures round-trip correctly."""
        row = _sample_row()
        row["data"] = {"list": [1, 2, 3], "nested": {"deep": {"value": True}}}
        sqlite_backend.upsert("roundtrip_test", row)
        result = sqlite_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None
        assert result["data"] == row["data"]

    def test_decimal_serialization(self, sqlite_backend: SQLiteBackend) -> None:
        """Decimal values serialize through the serialize_value path."""
        val = Decimal("3.14")
        serialized = sqlite_backend.serialize_value(val, "REAL")
        assert serialized == pytest.approx(3.14)

    def test_list_serialization(self, sqlite_backend: SQLiteBackend) -> None:
        """List values serialize/deserialize through JSON."""
        val = [0.1, 0.2, 0.3]
        serialized = sqlite_backend.serialize_value(val, "TEXT_ARRAY")
        assert isinstance(serialized, str)
        deserialized = sqlite_backend.deserialize_field(serialized, "TEXT_ARRAY")
        assert deserialized == val

    def test_uuid_utils_if_available(self, sqlite_backend: SQLiteBackend) -> None:
        """uuid_utils.UUID values serialize correctly if available."""
        try:
            from uuid_utils import UUID as UuidUtilsUUID
        except ImportError:
            pytest.skip("uuid_utils not installed")

        test_uuid = UuidUtilsUUID(str(uuid.uuid4()))
        serialized = sqlite_backend.serialize_value(test_uuid, "TEXT_UUID")
        assert isinstance(serialized, str)
        assert serialized == str(test_uuid)


class TestDuckDBRoundTrip:
    """Write sample data to DuckDB L1 and read it back."""

    @pytest.fixture()
    def duckdb_backend(self) -> Any:
        """Create and initialize a DuckDBBackend if available."""
        duckdb = pytest.importorskip("duckdb")  # noqa: F841
        from threetears.core.cache.duckdb import DuckDBBackend

        b = DuckDBBackend()
        b.initialize(_make_comprehensive_metadata())
        yield b
        b.reset()

    def test_write_and_readback(self, duckdb_backend: Any) -> None:
        """Write sample data and verify readback matches."""
        row = _sample_row()
        duckdb_backend.upsert("roundtrip_test", row)

        result = duckdb_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None, "Row not found after upsert"

        assert result["name"] == row["name"]
        assert result["count"] == row["count"]
        assert result["active"] is True
        assert result["data"] == row["data"]
        assert result["raw_bytes"] == row["raw_bytes"]

    def test_null_values(self, duckdb_backend: Any) -> None:
        """NULL values round-trip through DuckDB."""
        row = _sample_row()
        row["name"] = None
        row["data"] = None
        row["created_at"] = None
        duckdb_backend.upsert("roundtrip_test", row)
        result = duckdb_backend.select_by_id("roundtrip_test", row["id"])
        assert result is not None
        assert result["name"] is None
        assert result["data"] is None
