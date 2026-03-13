"""Protocol compliance tests for L1 cache backends.

Parametrized to run against both SQLite and DuckDB backends.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB, UUID

from threetears.core.cache.base import L1Backend


def _make_metadata() -> MetaData:
    """Create a test SQLAlchemy MetaData with a single table."""
    metadata = MetaData()
    Table(
        "test_entities",
        metadata,
        Column("id", UUID, primary_key=True),
        Column("name", String(255)),
        Column("age", Integer),
        Column("active", Boolean),
        Column("data", JSONB),
        Column("created_at", DateTime),
    )
    return metadata


@pytest.fixture(params=["sqlite", "duckdb"])
def backend(request: pytest.FixtureRequest) -> L1Backend:
    """Create and initialize a backend, reset after test."""
    if request.param == "sqlite":
        from threetears.core.cache.sqlite import SQLiteBackend

        b = SQLiteBackend(db_name=f"test_{uuid.uuid4().hex[:8]}")
    elif request.param == "duckdb":
        duckdb = pytest.importorskip("duckdb")  # noqa: F841
        from threetears.core.cache.duckdb import DuckDBBackend

        b = DuckDBBackend()
    else:
        pytest.fail(f"Unknown backend: {request.param}")

    metadata = _make_metadata()
    b.initialize(metadata)

    yield b

    b.reset()


def _sample_row(entity_id: str | None = None) -> dict:
    """Return a sample row dict for test_entities."""
    return {
        "id": entity_id or str(uuid.uuid4()),
        "name": "Alice",
        "age": 30,
        "active": True,
        "data": {"role": "admin", "tags": ["a", "b"]},
        "created_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    }


class TestProtocolCompliance:
    """Verify both backends satisfy the L1Backend protocol."""

    def test_isinstance_check(self, backend: L1Backend) -> None:
        assert isinstance(backend, L1Backend)

    def test_is_initialized(self, backend: L1Backend) -> None:
        assert backend.is_initialized() is True


class TestInitialize:
    """Verify initialize creates tables."""

    def test_initialize_creates_tables(self, backend: L1Backend) -> None:
        # After initialize, we should be able to select from the table
        results = backend.execute_query("SELECT * FROM test_entities")
        assert results == []

    def test_initialize_is_idempotent(self, backend: L1Backend) -> None:
        metadata = _make_metadata()
        # Should not raise
        backend.initialize(metadata)
        assert backend.is_initialized() is True


class TestUpsertAndSelect:
    """Test upsert + select_by_id round-trip."""

    def test_upsert_and_select_round_trip(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)
        result = backend.select_by_id("test_entities", row["id"])

        assert result is not None
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["active"] is True
        assert result["data"] == {"role": "admin", "tags": ["a", "b"]}

    def test_upsert_updates_existing_row(self, backend: L1Backend) -> None:
        entity_id = str(uuid.uuid4())
        row = _sample_row(entity_id)
        backend.upsert("test_entities", row)

        # Update
        row["name"] = "Bob"
        row["age"] = 42
        backend.upsert("test_entities", row)

        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["name"] == "Bob"
        assert result["age"] == 42

    def test_select_by_id_returns_none_for_missing(self, backend: L1Backend) -> None:
        result = backend.select_by_id("test_entities", str(uuid.uuid4()))
        assert result is None


class TestSelectBatch:
    """Test select_batch returns correct subset."""

    def test_select_batch(self, backend: L1Backend) -> None:
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, eid in enumerate(ids):
            row = _sample_row(eid)
            row["name"] = f"User{i}"
            backend.upsert("test_entities", row)

        # Select first two
        results = backend.select_batch("test_entities", ids[:2])
        assert len(results) == 2
        result_ids = {r["id"] for r in results}
        # Compare as strings since some backends may return UUID objects
        assert {str(rid) for rid in result_ids} == {ids[0], ids[1]}

    def test_select_batch_empty_list(self, backend: L1Backend) -> None:
        results = backend.select_batch("test_entities", [])
        assert results == []


class TestDelete:
    """Test delete_by_id removes entry."""

    def test_delete_by_id(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        backend.delete_by_id("test_entities", row["id"])
        result = backend.select_by_id("test_entities", row["id"])
        assert result is None

    def test_delete_nonexistent_is_noop(self, backend: L1Backend) -> None:
        # Should not raise
        backend.delete_by_id("test_entities", str(uuid.uuid4()))


class TestReset:
    """Test reset clears state."""

    def test_reset_clears_state(self, backend: L1Backend) -> None:
        assert backend.is_initialized() is True
        backend.reset()
        assert backend.is_initialized() is False
