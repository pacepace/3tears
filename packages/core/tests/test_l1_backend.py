"""Protocol compliance tests for L1 cache backends.

Parametrized to run against both SQLite and DuckDB backends.
"""

from __future__ import annotations

import uuid
from pathlib import Path
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


class TestProjectedReads:
    """Test select_by_id and select_batch column projection."""

    def test_select_by_id_columns_returns_only_requested(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        result = backend.select_by_id("test_entities", row["id"], columns=["name", "age"])

        assert result is not None
        assert set(result.keys()) == {"name", "age"}
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_select_by_id_columns_deserializes_requested(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        result = backend.select_by_id("test_entities", row["id"], columns=["data", "active"])

        assert result is not None
        assert result["data"] == {"role": "admin", "tags": ["a", "b"]}
        assert result["active"] is True

    def test_select_by_id_columns_missing_row_returns_none(self, backend: L1Backend) -> None:
        result = backend.select_by_id("test_entities", str(uuid.uuid4()), columns=["name"])
        assert result is None

    def test_select_by_id_columns_default_returns_all(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        result = backend.select_by_id("test_entities", row["id"])

        assert result is not None
        assert set(result.keys()) == set(row.keys())

    def test_select_by_id_columns_unknown_column_raises(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        with pytest.raises(ValueError, match="no_such_column"):
            backend.select_by_id("test_entities", row["id"], columns=["no_such_column"])

    def test_select_by_id_columns_empty_raises(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        with pytest.raises(ValueError, match="columns"):
            backend.select_by_id("test_entities", row["id"], columns=[])

    def test_select_by_id_columns_duplicates_collapse(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        result = backend.select_by_id("test_entities", row["id"], columns=["name", "name"])

        assert result is not None
        assert set(result.keys()) == {"name"}
        assert result["name"] == "Alice"

    def test_select_batch_columns_returns_only_requested(self, backend: L1Backend) -> None:
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, eid in enumerate(ids):
            row = _sample_row(eid)
            row["name"] = f"User{i}"
            backend.upsert("test_entities", row)

        results = backend.select_batch("test_entities", ids[:2], columns=["id", "name"])

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {"id", "name"}
        names = {r["name"] for r in results}
        assert names == {"User0", "User1"}

    def test_select_batch_columns_deserializes_requested(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        results = backend.select_batch("test_entities", [row["id"]], columns=["data"])

        assert len(results) == 1
        assert results[0]["data"] == {"role": "admin", "tags": ["a", "b"]}

    def test_select_batch_columns_unknown_column_raises(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        with pytest.raises(ValueError, match="no_such_column"):
            backend.select_batch("test_entities", [row["id"]], columns=["no_such_column"])

    def test_select_batch_columns_empty_raises(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)

        with pytest.raises(ValueError, match="columns"):
            backend.select_batch("test_entities", [row["id"]], columns=[])


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


class TestBulkWrites:
    """``upsert_many`` writes as ``upsert`` would row by row, on every backend."""

    def test_rows_round_trip(self, backend: L1Backend) -> None:
        rows = [_sample_row(), {**_sample_row(), "name": "Bob", "age": 41}]
        assert backend.upsert_many("test_entities", rows) == 2
        for row in rows:
            got = backend.select_by_id("test_entities", row["id"])
            assert got is not None
            assert (got["name"], got["age"], got["data"]) == (row["name"], row["age"], row["data"])

    def test_an_existing_row_is_updated(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert("test_entities", row)
        backend.upsert_many("test_entities", [{**row, "name": "Carol"}])
        got = backend.select_by_id("test_entities", row["id"])
        assert got is not None and got["name"] == "Carol"

    def test_no_rows_writes_nothing(self, backend: L1Backend) -> None:
        assert backend.upsert_many("test_entities", []) == 0

    def test_ragged_rows_are_refused(self, backend: L1Backend) -> None:
        full, partial = _sample_row(), {"id": str(uuid.uuid4()), "name": "Dan"}
        with pytest.raises(ValueError, match="row 1 names different columns"):
            backend.upsert_many("test_entities", [full, partial])

    def test_column_types_are_public(self, backend: L1Backend) -> None:
        types = backend.column_types("test_entities")
        assert set(types) >= {"id", "name", "age", "active", "data", "created_at"}
        assert backend.column_types("no_such_table") == {}


def test_duckdb_loads_a_parquet_file(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    from sqlalchemy import BigInteger

    from threetears.core.cache.duckdb import DuckDBBackend

    metadata = MetaData()
    Table(
        "results",
        metadata,
        Column("row_key", BigInteger, primary_key=True),
        Column("state", String(2)),
        Column("votes", Integer),
        Column("note", String(20)),
    )
    backend = DuckDBBackend()
    backend.initialize(metadata)
    # the file has a column the table does not declare, and lacks one it does
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES ('TX', 10, 1), ('GA', 7, 2)) AS t(state, votes, extra)) TO "
        f"'{tmp_path / 'r.parquet'}' (FORMAT parquet)"
    )
    try:
        assert backend.load_parquet("results", tmp_path / "r.parquet", row_number_column="row_key") == 2
        rows = backend.execute_query("SELECT row_key, state, votes, note FROM results ORDER BY row_key")
        assert rows == [
            {"row_key": 0, "state": "TX", "votes": 10, "note": None},
            {"row_key": 1, "state": "GA", "votes": 7, "note": None},
        ]
        with pytest.raises(ValueError, match="unknown table"):
            backend.load_parquet("missing", tmp_path / "r.parquet")
    finally:
        backend.reset()


class TestBulkWriteEdges:
    """the corners a bulk write must still treat as ``upsert`` would row by row."""

    def test_a_key_repeated_in_a_batch_keeps_its_last_row(self, backend: L1Backend) -> None:
        row = _sample_row()
        backend.upsert_many("test_entities", [{**row, "name": "first"}, {**row, "name": "last"}])
        got = backend.select_by_id("test_entities", row["id"])
        assert got is not None and got["name"] == "last"

    def test_a_key_given_as_a_uuid_and_as_text_is_one_key(self, backend: L1Backend) -> None:
        row = _sample_row()
        as_text, as_uuid = {**row, "name": "text"}, {**row, "id": uuid.UUID(row["id"]), "name": "uuid"}
        backend.upsert_many("test_entities", [as_text, as_uuid])
        got = backend.select_by_id("test_entities", row["id"])
        assert got is not None and got["name"] == "uuid"

    def test_column_types_are_the_tables_own_columns(self, backend: L1Backend) -> None:
        # the backend's own cache-stamp column is not one of the table's
        assert set(backend.column_types("test_entities")) == {"id", "name", "age", "active", "data", "created_at"}


def test_a_failed_sqlite_write_leaves_no_transaction_open() -> None:
    """a bad row rolls back, so the thread's next write is not refused by its own open transaction."""
    from threetears.core.cache.sqlite import SQLiteBackend

    backend = SQLiteBackend(db_name=f"test_{uuid.uuid4().hex[:8]}")
    backend.initialize(_make_metadata())
    try:
        good = _sample_row()
        backend.upsert("test_entities", good)
        # a row the database refuses: two rows in one batch, the second missing its key
        with pytest.raises(Exception):  # noqa: B017 -- any refusal will do; the point is what follows
            backend.upsert_many("test_entities", [_sample_row(), {**_sample_row(), "id": None}])
        later = _sample_row()
        backend.upsert("test_entities", later)
        assert backend.select_by_id("test_entities", later["id"]) is not None
    finally:
        backend.reset()


def test_a_write_sqlite_rolled_back_itself_raises_its_own_error() -> None:
    """a full disk makes SQLite roll the transaction back itself; the error is the full disk, not the rollback."""
    import sqlite3

    from threetears.core.cache.sqlite import SQLiteBackend

    backend = SQLiteBackend(db_name=f"test_{uuid.uuid4().hex[:8]}")
    backend.initialize(_make_metadata())
    try:
        conn = backend.get_connection()
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count = {pages + 2}")
        with pytest.raises(sqlite3.OperationalError, match="full"):
            backend.upsert_many("test_entities", [{**_sample_row(), "name": "x" * 4000} for _ in range(200)])
        assert not conn.in_transaction
    finally:
        backend.reset()


def test_a_sqlite_table_of_only_key_columns_takes_an_upsert() -> None:
    from threetears.core.cache.sqlite import SQLiteBackend

    metadata = MetaData()
    Table("tags", metadata, Column("name", String(20), primary_key=True))
    backend = SQLiteBackend(db_name=f"test_{uuid.uuid4().hex[:8]}")
    backend.initialize(metadata)
    try:
        backend.upsert("tags", {"name": "a"}, primary_key="name")
        backend.upsert("tags", {"name": "a"}, primary_key="name")
        assert backend.execute_query("SELECT name FROM tags") == [{"name": "a"}]
    finally:
        backend.reset()


@pytest.mark.parametrize("kind", ["sqlite", "duckdb"])
def test_names_with_spaces_and_capitals_survive_every_statement(kind: str) -> None:
    """a table and key column named as a converted extract names them: created, written, read, deleted."""
    if kind == "duckdb":
        pytest.importorskip("duckdb")
        from threetears.core.cache.duckdb import DuckDBBackend

        backend: L1Backend = DuckDBBackend()
    else:
        from threetears.core.cache.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_name=f"test_{uuid.uuid4().hex[:8]}")
    metadata = MetaData()
    Table(
        "Results by County",
        metadata,
        Column("County Id", String(20), primary_key=True),
        Column("% of Exp. In", String(20)),
    )
    backend.initialize(metadata)
    try:
        backend.upsert("Results by County", {"County Id": "51001", "% of Exp. In": "0.55"}, primary_key="County Id")
        backend.upsert_many(
            "Results by County", [{"County Id": "51003", "% of Exp. In": "0.99"}], primary_key="County Id"
        )
        assert backend.select_by_id(
            "Results by County", "51001", primary_key="County Id", columns=["% of Exp. In"]
        ) == {"% of Exp. In": "0.55"}
        found = backend.select_batch("Results by County", ["51001", "51003"], primary_key="County Id")
        assert {row["County Id"] for row in found} == {"51001", "51003"}
        backend.delete_by_id("Results by County", "51001", primary_key="County Id")
        assert backend.select_by_id("Results by County", "51001", primary_key="County Id") is None
    finally:
        backend.reset()
