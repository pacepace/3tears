"""DuckDB-specific tests for the L1 cache backend."""

from __future__ import annotations

from typing import Any

import importlib
import sys
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

    def test_import_error_without_duckdb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A process without duckdb imports the module, and is told what to install on use.

        The module is imported afresh with ``duckdb`` unimportable, which is exactly the state
        an install without the extra is in. ``patch.dict`` restores ``sys.modules`` afterwards,
        and the package attribute the fresh import rebinds is restored by ``monkeypatch``, so
        nothing else in the process sees the duckdb-less copy.
        """
        import threetears.core.cache as cache_pkg

        monkeypatch.setattr(cache_pkg, "duckdb", cache_pkg.duckdb)
        with mock.patch.dict("sys.modules", {"duckdb": None}):
            sys.modules.pop("threetears.core.cache.duckdb", None)
            without_duckdb = importlib.import_module("threetears.core.cache.duckdb")

            assert without_duckdb.DuckDBBackend is not DuckDBBackend, "the module was not imported afresh"
            with pytest.raises(ImportError, match="duckdb"):
                without_duckdb.DuckDBBackend()


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


class TestUpsertFiltersToTheRegisteredSchema:
    """A framework-injected column must not reach the SQL against a table without it.

    The collection's pull-through stamps every row with the L1 cache-age column
    before writing, and it does so backend-agnostically. DuckDB declares no such
    column, so an unfiltered write failed on every pull-through against a table
    that is otherwise fine. SQLiteBackend had always filtered; this backend had
    not, and nothing noticed because nothing constructs it outside tests.
    """

    def test_any_undeclared_key_is_dropped_too(self, backend: DuckDBBackend) -> None:
        """The filter is general, not a special case for the stamp."""
        entity_id = uuid.uuid4()
        backend.upsert(
            "test_entities",
            {"id": entity_id, "name": "kept", "not_a_column": "dropped"},
            "id",
        )

        row = backend.select_by_id("test_entities", entity_id, "id")
        assert row is not None
        assert row["name"] == "kept"
        assert "not_a_column" not in row

    def test_declared_columns_still_round_trip(self, backend: DuckDBBackend) -> None:
        """The control: filtering must not have narrowed a legitimate write."""
        entity_id = uuid.uuid4()
        backend.upsert("test_entities", {"id": entity_id, "name": "alice", "age": 30}, "id")

        row = backend.select_by_id("test_entities", entity_id, "id")
        assert row is not None
        assert row["name"] == "alice"
        assert row["age"] == 30


def _partitioned_backend() -> DuckDBBackend:
    metadata = MetaData()
    Table(
        "results",
        metadata,
        Column("race", String(64), primary_key=True),
        Column("county", String(64), primary_key=True),
        Column("state", String(8)),
        Column("votes", Integer),
        Column("counted_at", DateTime(timezone=True)),
    )
    backend = DuckDBBackend()
    backend.initialize(metadata)
    return backend


def _result(race: str, county: str, state: str | None, votes: int) -> dict[str, object]:
    return {
        "race": race,
        "county": county,
        "state": state,
        "votes": votes,
        "counted_at": datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc),
    }


class TestPartitions:
    """one scope of a table at a time: exported as Arrow, replaced whole in one transaction."""

    def test_a_partition_exports_as_arrow_in_key_order(self) -> None:
        pytest.importorskip("pyarrow")
        backend = _partitioned_backend()
        backend.upsert_many(
            "results",
            [_result("r1", "c2", "TX", 2), _result("r1", "c1", "TX", 1), _result("r1", "c9", "CA", 9)],
            ("race", "county"),
        )
        exported = backend.export_partition("results", "state", "TX", order_by=("race", "county"))
        assert exported.num_rows == 2
        assert exported.column("county").to_pylist() == ["c1", "c2"]
        assert backend.export_partition("results", "state", None, order_by=("race",)).num_rows == 0

    def test_a_bulk_insert_converts_its_rows_once_not_value_by_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """binding rows as Python lists makes DuckDB try to import pandas once per value, a search of
        the whole import path each time where pandas is absent: seconds for a few thousand rows, and
        a rebuild that never finishes on a starved host. The rows go through one Arrow table instead."""
        pytest.importorskip("pyarrow")
        import importlib.machinery

        from threetears.core.cache.duckdb import PartitionReplacement

        backend = _partitioned_backend()
        backend.replace_partitions(  # warm: pyarrow's own first import is not the subject
            [
                PartitionReplacement(
                    table="results",
                    column="state",
                    value="CA",
                    rows=[_result("r", "c", "CA", 1)],
                    primary_key=("race", "county"),
                )
            ]
        )
        searches = [0]
        real = importlib.machinery.PathFinder.find_spec

        def counting(name: str, path: Any = None, target: Any = None) -> Any:
            searches[0] += 1
            return real(name, path, target)

        monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", staticmethod(counting))
        rows = [_result("r1", f"c{i:04d}", "TX", i) for i in range(2000)]

        backend.replace_partitions(
            [
                PartitionReplacement(
                    table="results", column="state", value="TX", rows=rows, primary_key=("race", "county")
                )
            ]
        )

        assert searches[0] < 20, f"{searches[0]} import-path searches for one bulk insert of 2000 rows"
        assert backend.export_partition("results", "state", "TX", order_by=("race", "county")).num_rows == 2000

    def test_a_column_arrow_cannot_type_as_one_still_round_trips_exactly(self) -> None:
        """mixed kinds in one column, and an int past 64 bits, fall back to binding the values; the
        rows must come back exactly as written either way."""
        pytest.importorskip("pyarrow")
        from sqlalchemy import Numeric

        metadata = MetaData()
        Table(
            "mixed",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("label", String(32)),
            Column("big", Numeric(30, 0)),
        )
        backend = DuckDBBackend()
        backend.initialize(metadata)
        rows = [
            {"id": 1, "label": "a", "big": 2**70},
            {"id": 2, "label": 7, "big": 3},  # an int where the others are text
        ]

        backend.upsert_many("mixed", rows, "id")

        held = backend.execute_query("SELECT id, label, big FROM mixed ORDER BY id")
        assert [(r["id"], r["label"], int(r["big"])) for r in held] == [(1, "a", 2**70), (2, "7", 3)]

    def test_a_read_holds_the_state_from_when_it_began_not_from_its_first_query(self) -> None:
        pytest.importorskip("pyarrow")
        from threetears.core.cache.duckdb import PartitionReplacement

        backend = _partitioned_backend()
        key = ("race", "county")
        backend.upsert_many("results", [_result("r1", "c1", "TX", 1)], key)

        with backend.read_snapshot() as cursor:
            backend.replace_partitions(
                [
                    PartitionReplacement(
                        table="results",
                        column="state",
                        value="TX",
                        rows=[_result("r1", "c1", "TX", 2)],
                        primary_key=key,
                    )
                ]
            )
            assert cursor.execute("SELECT votes FROM results").fetchall() == [(1,)], "the read saw a later commit"

    def test_rows_export_as_the_partition_would_hold_them_and_change_nothing(self) -> None:
        pytest.importorskip("pyarrow")
        from threetears.core.cache.duckdb import PartitionReplacement

        backend = _partitioned_backend()
        key = ("race", "county")
        backend.upsert_many("results", [_result("r1", "c1", "TX", 1), _result("r1", "c9", "CA", 9)], key)
        fresh = [_result("r2", "c3", "TX", 30), _result("r1", "c2", "TX", 20)]

        staged = backend.export_rows("results", "state", "TX", fresh, primary_key=key, order_by=key)

        assert staged.column("county").to_pylist() == ["c2", "c3"]
        held = backend.execute_query("SELECT county FROM results ORDER BY county")
        assert held == [{"county": "c1"}, {"county": "c9"}], "an export changed the table"
        backend.replace_partitions(
            [PartitionReplacement(table="results", column="state", value="TX", rows=fresh, primary_key=key)]
        )
        assert staged.equals(backend.export_partition("results", "state", "TX", order_by=key))
        assert backend.export_rows("results", "state", "TX", [], primary_key=key, order_by=key).num_rows == 0

    def test_replacing_partitions_swaps_each_scope_whole_and_leaves_the_rest(self) -> None:
        pytest.importorskip("pyarrow")
        from threetears.core.cache.duckdb import PartitionReplacement

        source = _partitioned_backend()
        source.upsert_many(
            "results", [_result("r1", "c1", "TX", 10), _result("r2", "c1", "TX", 20)], ("race", "county")
        )
        chunk = source.export_partition("results", "state", "TX", order_by=("race", "county"))

        backend = _partitioned_backend()
        backend.upsert_many(
            "results",
            [_result("r1", "c1", "TX", 1), _result("r9", "c9", "TX", 9), _result("r1", "c5", "CA", 5)],
            ("race", "county"),
        )
        written = backend.replace_partitions(
            [
                PartitionReplacement(table="results", column="state", value="TX", arrow=chunk),
                PartitionReplacement(
                    table="results",
                    column="state",
                    value="DE",
                    rows=[_result("r3", "c3", "DE", 3)],
                    primary_key=("race", "county"),
                ),
            ]
        )
        assert written == 3
        rows = backend.execute_query("SELECT race, state, votes FROM results ORDER BY state, race")
        assert rows == [
            {"race": "r1", "state": "CA", "votes": 5},
            {"race": "r3", "state": "DE", "votes": 3},
            {"race": "r1", "state": "TX", "votes": 10},
            {"race": "r2", "state": "TX", "votes": 20},
        ]

    def test_a_failing_replacement_changes_nothing(self) -> None:
        pytest.importorskip("pyarrow")
        from threetears.core.cache.duckdb import PartitionReplacement

        backend = _partitioned_backend()
        backend.upsert_many("results", [_result("r1", "c1", "TX", 1)], ("race", "county"))
        with pytest.raises(ValueError):
            backend.replace_partitions(
                [
                    PartitionReplacement(table="results", column="state", value="TX", rows=[]),
                    PartitionReplacement(table="nope", column="state", value="TX", rows=[]),
                ]
            )
        assert backend.execute_query("SELECT count(*) AS n FROM results") == [{"n": 1}]

    def test_an_empty_replacement_removes_the_scope(self) -> None:
        from threetears.core.cache.duckdb import PartitionReplacement

        backend = _partitioned_backend()
        backend.upsert_many("results", [_result("r1", "c1", "TX", 1), _result("r1", "c2", None, 2)], ("race", "county"))
        backend.replace_partitions([PartitionReplacement(table="results", column="state", value=None, rows=[])])
        assert backend.execute_query("SELECT county FROM results") == [{"county": "c1"}]

    def test_a_read_snapshot_holds_still_across_a_replacement(self) -> None:
        from threetears.core.cache.duckdb import PartitionReplacement

        backend = _partitioned_backend()
        backend.upsert_many("results", [_result("r1", "c1", "TX", 1)], ("race", "county"))
        with backend.read_snapshot() as cursor:
            assert cursor.execute("SELECT votes FROM results").fetchall() == [(1,)]
            backend.replace_partitions(
                [
                    PartitionReplacement(
                        table="results",
                        column="state",
                        value="TX",
                        rows=[_result("r1", "c1", "TX", 2)],
                        primary_key=("race", "county"),
                    )
                ]
            )
            assert cursor.execute("SELECT votes FROM results").fetchall() == [(1,)]
        with backend.read_snapshot() as cursor:
            assert cursor.execute("SELECT votes FROM results").fetchall() == [(2,)]

    def test_the_schema_digest_names_the_columns_and_their_types(self) -> None:
        assert _partitioned_backend().schema_digest("results") == _partitioned_backend().schema_digest("results")
        other = DuckDBBackend()
        metadata = MetaData()
        Table("results", metadata, Column("race", String(64), primary_key=True), Column("votes", String(8)))
        other.initialize(metadata)
        assert other.schema_digest("results") != _partitioned_backend().schema_digest("results")
