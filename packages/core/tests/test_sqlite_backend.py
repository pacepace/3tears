"""SQLite-specific tests for the L1 cache backend."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal

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

from threetears.core.cache.sqlite import SQLiteBackend


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


@pytest.fixture()
def backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_sqlite_{uuid.uuid4().hex[:8]}")
    metadata = _make_metadata()
    b.initialize(metadata)
    yield b
    b.reset()


class TestThreadLocalConnections:
    """Verify each thread gets a different connection object."""

    def test_different_threads_get_different_connections(self, backend: SQLiteBackend) -> None:
        connections: list[object] = []
        barrier = threading.Barrier(2)

        def _get_conn() -> None:
            conn = backend.get_connection()
            connections.append(conn)
            barrier.wait()

        t1 = threading.Thread(target=_get_conn)
        t2 = threading.Thread(target=_get_conn)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(connections) == 2
        # distinct pooled proxies, each wrapping a different thread-local
        # sqlite3.Connection; the proxy's ``__getattr__`` delegates
        # ``execute`` to the underlying connection's builtin bound method,
        # whose ``__self__`` is the underlying Connection instance.
        # comparing ``execute.__self__`` across proxies proves the
        # thread-local distinctness without reaching into the proxy's
        # private wrapped-connection slot.
        assert connections[0].execute.__self__ is not connections[1].execute.__self__


class TestGetConnectionBeforeInitialize:
    """Verify get_connection raises before initialize."""

    def test_raises_runtime_error(self) -> None:
        b = SQLiteBackend(db_name="uninit_test")
        with pytest.raises(RuntimeError, match="not initialized"):
            b.get_connection()


class TestExecuteQuery:
    """Verify execute_query works for generic SELECT."""

    def test_execute_query(self, backend: SQLiteBackend) -> None:
        entity_id = str(uuid.uuid4())
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "Query Test",
                "age": 25,
                "active": True,
                "data": None,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        results = backend.execute_query("SELECT name, age FROM test_entities WHERE age = ?", (25,))
        assert len(results) == 1
        assert results[0]["name"] == "Query Test"
        assert results[0]["age"] == 25


class TestUpsertFiltersUnknownColumns:
    """``upsert`` must drop keys the table schema doesn't declare.

    ``BaseCollection.save_entity`` unconditionally injects
    ``date_created`` / ``date_updated`` for new entities, but not every
    entity's table carries those columns (e.g. ``agent_skill_invocations``
    uses ``invoked_at`` and has neither). Writing an unknown column to
    SQLite raises ``OperationalError: table X has no column named ...``;
    the L1 write must mirror the L3 projection and silently drop unknown
    keys instead.
    """

    def test_unknown_columns_are_dropped(self, backend: SQLiteBackend) -> None:
        entity_id = str(uuid.uuid4())
        # ``date_created`` / ``date_updated`` are NOT columns on
        # ``test_entities``; the framework would inject them on a new
        # entity. The upsert must not raise.
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "Filter Test",
                "age": 30,
                "active": False,
                "data": None,
                "created_at": None,
                "raw_bytes": None,
                "date_created": datetime.now(timezone.utc),
                "date_updated": datetime.now(timezone.utc),
            },
        )
        row = backend.select_by_id("test_entities", entity_id)
        assert row is not None
        assert row["name"] == "Filter Test"
        # The unknown keys never reached the row.
        assert "date_created" not in row
        assert "date_updated" not in row


class TestSerializationRoundTrip:
    """Verify round-trip serialization for various Python types."""

    def test_uuid_round_trip(self, backend: SQLiteBackend) -> None:
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

    def test_datetime_round_trip(self, backend: SQLiteBackend) -> None:
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

    def test_json_dict_round_trip(self, backend: SQLiteBackend) -> None:
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

    def test_bool_round_trip(self, backend: SQLiteBackend) -> None:
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

        # Also test False
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "bool test",
                "age": 1,
                "active": False,
                "data": None,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["active"] is False

    def test_decimal_round_trip(self, backend: SQLiteBackend) -> None:
        """Decimal is serialized to float for REAL columns."""
        val = Decimal("3.14")
        serialized = backend.serialize_value(val, "REAL")
        assert serialized == pytest.approx(3.14)

    def test_bytes_round_trip(self, backend: SQLiteBackend) -> None:
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

    def test_list_round_trip(self, backend: SQLiteBackend) -> None:
        """Lists serialized via serialize_value become JSON strings."""
        val = [1, 2, 3]
        serialized = backend.serialize_value(val, "TEXT_ARRAY")
        assert serialized == "[1, 2, 3]"
        deserialized = backend.deserialize_field(serialized, "TEXT_ARRAY")
        assert deserialized == [1, 2, 3]

    def test_generic_largebinary_round_trip(self) -> None:
        """A generic ``sqlalchemy.LargeBinary`` column must round-trip bytes
        losslessly through SQLite (proving it maps to ``TEXT_BYTEA``).

        Regression: ``webhook_subscriptions.secret_ciphertext`` is declared
        with the generic ``sqlalchemy.LargeBinary`` (not the postgresql
        ``BYTEA`` dialect type). The mapper previously only matched
        ``PgBYTEA``, so generic ``LargeBinary`` fell through to plain
        ``TEXT``: bytes were written as a hex string but never decoded back
        to ``bytes`` on read, crashing ``save_entity`` downstream with
        ``TypeError: string argument without an encoding``. A clean
        byte-for-byte round trip here proves the ``TEXT_BYTEA`` mapping.
        """
        metadata = MetaData()
        Table(
            "lb_entities",
            metadata,
            Column("id", UUID, primary_key=True),
            Column("blob", LargeBinary),
        )
        b = SQLiteBackend(db_name=f"test_lb_{uuid.uuid4().hex[:8]}")
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
        """The postgresql ``BYTEA`` dialect type must still round-trip
        bytes losslessly (no regression from broadening the mapper's check
        to the ``LargeBinary`` base class — ``BYTEA`` subclasses
        ``LargeBinary``).
        """
        metadata = MetaData()
        Table(
            "pg_bytea_entities",
            metadata,
            Column("id", UUID, primary_key=True),
            Column("blob", BYTEA),
        )
        b = SQLiteBackend(db_name=f"test_pgbytea_{uuid.uuid4().hex[:8]}")
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

    def test_none_round_trip(self, backend: SQLiteBackend) -> None:
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
