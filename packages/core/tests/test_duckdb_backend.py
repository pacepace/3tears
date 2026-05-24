"""DuckDB-specific tests for the L1 cache backend."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest import mock

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID


def _make_metadata() -> MetaData:
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
        Column("raw_bytes", BYTEA),
    )
    return metadata


# Skip all tests in this module if duckdb is not installed
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402


@pytest.fixture()
def backend() -> DuckDBBackend:
    b = DuckDBBackend()
    metadata = _make_metadata()
    b.initialize(metadata)
    yield b
    b.reset()


class TestImportError:
    """Verify ImportError when duckdb is not installed."""

    def test_import_error_without_duckdb(self) -> None:
        with mock.patch.dict("sys.modules", {"duckdb": None}):
            import threetears.core.cache.duckdb as duckdb_mod

            # Save original state
            orig_has = duckdb_mod._HAS_DUCKDB

            try:
                duckdb_mod._HAS_DUCKDB = False
                with pytest.raises(ImportError, match="duckdb"):
                    DuckDBBackend()
            finally:
                duckdb_mod._HAS_DUCKDB = orig_has


class TestThreadLocalConnections:
    """Verify thread-local connection behavior."""

    def test_different_threads_get_different_connections(self, backend: DuckDBBackend) -> None:
        connections: list[object] = []
        barrier = threading.Barrier(2)

        def _get_conn() -> None:
            conn = backend.get_connection()
            connections.append(id(conn))
            barrier.wait()

        t1 = threading.Thread(target=_get_conn)
        t2 = threading.Thread(target=_get_conn)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(connections) == 2
        assert connections[0] != connections[1]


class TestGetConnectionBeforeInitialize:
    """Verify get_connection raises before initialize."""

    def test_raises_runtime_error(self) -> None:
        b = DuckDBBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            b.get_connection()


class TestSerializationRoundTrip:
    """Verify round-trip serialization for various Python types."""

    def test_uuid_round_trip(self, backend: DuckDBBackend) -> None:
        entity_id = str(uuid.uuid4())
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "uuid test",
                "age": 1,
                "active": False,
                "data": None,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["id"] == uuid.UUID(entity_id)

    def test_datetime_round_trip(self, backend: DuckDBBackend) -> None:
        entity_id = str(uuid.uuid4())
        dt = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "dt test",
                "age": 1,
                "active": False,
                "data": None,
                "created_at": dt,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["created_at"] == dt

    def test_json_dict_round_trip(self, backend: DuckDBBackend) -> None:
        entity_id = str(uuid.uuid4())
        data = {"key": "value", "nested": {"a": 1}}
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "json test",
                "age": 1,
                "active": False,
                "data": data,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["data"] == data

    def test_bool_round_trip(self, backend: DuckDBBackend) -> None:
        entity_id = str(uuid.uuid4())
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "bool test",
                "age": 1,
                "active": True,
                "data": None,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["active"] is True

    def test_decimal_round_trip(self, backend: DuckDBBackend) -> None:
        val = Decimal("3.14")
        serialized = backend.serialize_value(val, "DOUBLE")
        assert serialized == pytest.approx(3.14)

    def test_bytes_round_trip(self, backend: DuckDBBackend) -> None:
        entity_id = str(uuid.uuid4())
        raw = b"\xde\xad\xbe\xef"
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "bytes test",
                "age": 1,
                "active": False,
                "data": None,
                "created_at": None,
                "raw_bytes": raw,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["raw_bytes"] == raw

    def test_generic_largebinary_round_trip(self) -> None:
        """A generic ``sqlalchemy.LargeBinary`` column must round-trip bytes
        losslessly through DuckDB (proving it maps to ``VARCHAR_BYTEA``).

        Regression: ``webhook_subscriptions.secret_ciphertext`` is declared
        with the generic ``sqlalchemy.LargeBinary`` (not the postgresql
        ``BYTEA`` dialect type). The L2 DuckDB mapper previously only matched
        ``PgBYTEA``, so generic ``LargeBinary`` fell through to plain
        ``VARCHAR``: bytes were written as a hex string but never decoded
        back to ``bytes`` on read, crashing ``save_entity`` downstream. A
        clean byte-for-byte round trip here proves the ``VARCHAR_BYTEA``
        mapping (parity with the SQLite L1 fix).
        """
        metadata = MetaData()
        Table(
            "lb_entities",
            metadata,
            Column("id", UUID, primary_key=True),
            Column("blob", LargeBinary),
        )
        b = DuckDBBackend()
        b.initialize(metadata)
        try:
            entity_id = str(uuid.uuid4())
            raw = b"\x00\x01\xfe\xff secret-bytes"
            b.upsert("lb_entities", {"id": entity_id, "blob": raw})
            result = b.select_by_id("lb_entities", entity_id)
            assert result is not None
            assert result["blob"] == raw
            assert isinstance(result["blob"], bytes)
        finally:
            b.reset()

    def test_pg_bytea_still_round_trips(self) -> None:
        """The postgresql ``BYTEA`` dialect type must still round-trip bytes
        losslessly (no regression from broadening the mapper's check to the
        ``LargeBinary`` base class — ``BYTEA`` subclasses ``LargeBinary``).
        """
        metadata = MetaData()
        Table(
            "pg_bytea_entities",
            metadata,
            Column("id", UUID, primary_key=True),
            Column("blob", BYTEA),
        )
        b = DuckDBBackend()
        b.initialize(metadata)
        try:
            entity_id = str(uuid.uuid4())
            raw = b"\xca\xfe\xba\xbe"
            b.upsert("pg_bytea_entities", {"id": entity_id, "blob": raw})
            result = b.select_by_id("pg_bytea_entities", entity_id)
            assert result is not None
            assert result["blob"] == raw
            assert isinstance(result["blob"], bytes)
        finally:
            b.reset()

    def test_list_round_trip(self, backend: DuckDBBackend) -> None:
        val = [1, 2, 3]
        serialized = backend.serialize_value(val, "VARCHAR_ARRAY")
        assert serialized == "[1, 2, 3]"
        deserialized = backend.deserialize_field(serialized, "VARCHAR_ARRAY")
        assert deserialized == [1, 2, 3]

    def test_none_round_trip(self, backend: DuckDBBackend) -> None:
        entity_id = str(uuid.uuid4())
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": None,
                "age": None,
                "active": None,
                "data": None,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["name"] is None
        assert result["age"] is None
        assert result["data"] is None
        assert result["created_at"] is None
        assert result["raw_bytes"] is None
