"""SQLite-specific tests for the L1 cache backend."""

from __future__ import annotations

import operator
import sqlite3
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Table,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID

from threetears.core.cache.base import UnsupportedL1ColumnError
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.schema_backed import (
    NUMERIC_TYPE,
    STRING_TYPE,
    TableSchema,
)
from threetears.core.collections.schema_backed import Column as SchemaColumn


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

    def test_json_values_encode_as_l2_encodes_them(self, backend: SQLiteBackend) -> None:
        """A JSON column holding a UUID, datetime or Decimal caches as L2 caches it, never raising.

        L2's payload encoder and L3's jsonb codec both write such a value as its
        string through ``schema_sql.json_default``. L1 used a bare ``json.dumps``
        and RAISED -- and it runs after the L3 commit, so a write both other tiers
        had taken was reported to its caller as a failure, and the invalidation
        broadcast after it never ran. A datetime is stored in the one form every
        tier writes (``serialization.json_datetime``).
        """
        entity_id = str(uuid.uuid4())
        nested_id = uuid.uuid4()
        when = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)
        data = {"answers": {"q1": {"session_id": nested_id, "date_answered": when, "score": Decimal("2.50")}}}
        backend.upsert(
            "test_entities",
            {
                "id": entity_id,
                "name": "json platform types",
                "age": 1,
                "active": False,
                "data": data,
                "created_at": None,
                "raw_bytes": None,
            },
        )
        result = backend.select_by_id("test_entities", entity_id)
        assert result is not None
        assert result["data"] == {
            "answers": {
                "q1": {
                    "session_id": str(nested_id),
                    "date_answered": "2026-10-01T12:30:00.000000+00:00",
                    "score": "2.50",
                }
            }
        }

    def test_json_values_of_no_platform_type_still_refuse(self, backend: SQLiteBackend) -> None:
        """Anything the other tiers' encoder refuses, L1 refuses too: the parity is exact, not a catch-all."""
        with pytest.raises(TypeError):
            backend.serialize_value({"opaque": object()}, "TEXT_JSON")

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


# parity-with: asyncpg.pgproto.pgproto.UUID
class _FakePgprotoUUID(uuid.UUID):
    """stands in for asyncpg's UUID subclass.

    asyncpg deserializes ``uuid`` columns to a C-implemented subclass of
    :class:`uuid.UUID`. sqlite3 adapter lookup is exact-type, so the
    subclass misses the ``register_adapter(uuid.UUID, str)`` registration
    and raises ``ProgrammingError`` when bound raw.
    """


class TestPkBindingSerialization:
    """issue #86: pk values are serialized at the L1 boundary like columns."""

    def _seeded_backend(self) -> tuple[SQLiteBackend, uuid.UUID]:
        b = SQLiteBackend(db_name=f"test_pkbind_{uuid.uuid4().hex[:8]}")
        b.initialize(_make_metadata())
        entity_id = uuid.uuid4()
        b.upsert("test_entities", {"id": entity_id, "name": "sprocket", "age": 1})
        return b, entity_id

    def test_select_by_id_accepts_uuid_subclass(self) -> None:
        b, entity_id = self._seeded_backend()
        try:
            result = b.select_by_id("test_entities", _FakePgprotoUUID(str(entity_id)))
            assert result is not None
            assert result["name"] == "sprocket"
        finally:
            b.reset()

    def test_select_batch_accepts_uuid_subclass(self) -> None:
        b, entity_id = self._seeded_backend()
        try:
            rows = b.select_batch("test_entities", [_FakePgprotoUUID(str(entity_id))])
            assert len(rows) == 1
            assert rows[0]["name"] == "sprocket"
        finally:
            b.reset()

    def test_delete_by_id_accepts_uuid_subclass(self) -> None:
        b, entity_id = self._seeded_backend()
        try:
            b.delete_by_id("test_entities", _FakePgprotoUUID(str(entity_id)))
            assert b.select_by_id("test_entities", entity_id) is None
        finally:
            b.reset()

    def test_select_by_id_accepts_datetime_pk(self) -> None:
        """datetime pks serialize at the read boundary like the write side."""
        metadata = MetaData()
        Table(
            "dt_keyed",
            metadata,
            Column("at", DateTime, primary_key=True),
            Column("name", String(255)),
        )
        b = SQLiteBackend(db_name=f"test_dtpk_{uuid.uuid4().hex[:8]}")
        b.initialize(metadata)
        try:
            at = datetime(2026, 6, 6, 12, 0, 0)
            b.upsert("dt_keyed", {"at": at, "name": "tick"}, primary_key="at")
            result = b.select_by_id("dt_keyed", at, primary_key="at")
            assert result is not None
            assert result["name"] == "tick"
        finally:
            b.reset()


class TestAdditiveInitialize:
    """initialize() registers new tables on subsequent calls."""

    def _table_metadata(self, table_name: str) -> MetaData:
        metadata = MetaData()
        Table(
            table_name,
            metadata,
            Column("id", String(255), primary_key=True),
            Column("name", String(255)),
        )
        return metadata

    def test_second_initialize_registers_new_table(self) -> None:
        b = SQLiteBackend(db_name=f"test_additive_{uuid.uuid4().hex[:8]}")
        b.initialize(self._table_metadata("first_table"))
        b.initialize(self._table_metadata("second_table"))
        try:
            b.upsert("first_table", {"id": "a", "name": "alpha"})
            b.upsert("second_table", {"id": "b", "name": "beta"})
            first = b.select_by_id("first_table", "a")
            second = b.select_by_id("second_table", "b")
            assert first is not None and first["name"] == "alpha"
            assert second is not None and second["name"] == "beta"
        finally:
            b.reset()

    def test_reinitialize_same_table_preserves_data(self) -> None:
        b = SQLiteBackend(db_name=f"test_reinit_{uuid.uuid4().hex[:8]}")
        b.initialize(self._table_metadata("stable_table"))
        try:
            b.upsert("stable_table", {"id": "a", "name": "alpha"})
            b.initialize(self._table_metadata("stable_table"))
            result = b.select_by_id("stable_table", "a")
            assert result is not None
            assert result["name"] == "alpha"
        finally:
            b.reset()

    def test_initialize_remains_one_shot_for_multi_table_metadata(self) -> None:
        """the established consumer shape (one metadata, all tables) still works."""
        metadata = MetaData()
        for name in ("multi_a", "multi_b"):
            Table(
                name,
                metadata,
                Column("id", String(255), primary_key=True),
                Column("name", String(255)),
            )
        b = SQLiteBackend(db_name=f"test_multi_{uuid.uuid4().hex[:8]}")
        b.initialize(metadata)
        try:
            b.upsert("multi_a", {"id": "a", "name": "alpha"})
            b.upsert("multi_b", {"id": "b", "name": "beta"})
            assert b.select_by_id("multi_a", "a") is not None
            assert b.select_by_id("multi_b", "b") is not None
        finally:
            b.reset()


class TestNumericReadsBackAsDecimal:
    """an exact NUMERIC(p, s) column is an INTEGER scaled by 10**s, read back as the Decimal L3 answers.

    The defect: the L1 stored NUMERIC as REAL and answered a float, so money read through L1 was a
    float, and ``Decimal * float`` raised wherever it was priced. Kept as a scaled integer, the
    column is exact, and compares, sorts and sums by value in SQL run against L1 directly.
    """

    def _backend(self) -> SQLiteBackend:
        metadata = MetaData()
        TableSchema(
            name="rates",
            primary_key="id",
            columns=[
                SchemaColumn("id", STRING_TYPE),
                SchemaColumn("price", NUMERIC_TYPE, precision=12, scale=2, nullable=True),
            ],
        ).to_sqlalchemy_table(metadata)
        Table("ratios", metadata, Column("id", String(255), primary_key=True), Column("ratio", Float()))
        b = SQLiteBackend(db_name=f"test_numeric_{uuid.uuid4().hex[:8]}")
        b.initialize(metadata)
        return b

    def test_a_numeric_column_reads_back_as_its_decimal_at_scale(self) -> None:
        b = self._backend()
        try:
            b.upsert("rates", {"id": "a", "price": Decimal("1.5")})
            b.upsert("rates", {"id": "b", "price": 2.125})
            b.upsert("rates", {"id": "c", "price": Decimal("-12345678.125")})
            a = b.select_by_id("rates", "a")
            assert a is not None
            assert isinstance(a["price"], Decimal) and str(a["price"]) == "1.50"
            assert a["price"] * Decimal(3) == Decimal("4.50")
            row_b = b.select_by_id("rates", "b")
            assert row_b is not None and str(row_b["price"]) == "2.13"
            # Postgres rounds half away from zero to the column's scale, negatives too
            row_c = b.select_by_id("rates", "c")
            assert row_c is not None and str(row_c["price"]) == "-12345678.13"
            assert b.column_types("rates")["price"] == "INTEGER_DECIMAL_2"
            b.upsert("ratios", {"id": "a", "ratio": 0.25})
            ratio = b.select_by_id("ratios", "a")
            # a float column stays a float: Postgres stores it as one, and asyncpg answers one
            assert ratio is not None and ratio["ratio"] == 0.25 and isinstance(ratio["ratio"], float)
        finally:
            b.reset()

    def test_sql_against_l1_compares_sorts_and_sums_by_value(self) -> None:
        """``'10.00' < '9.50'`` as text; as the scaled integers the column holds, 1000 > 950."""
        b = self._backend()
        try:
            for key, price in (("a", "9.50"), ("b", "10.00"), ("c", "-2.25"), ("d", "-10.00")):
                b.upsert("rates", {"id": key, "price": Decimal(price)})
            code = b.column_types("rates")["price"]
            ordered = b.execute_query('SELECT "id" FROM "rates" ORDER BY "price"')
            assert [row["id"] for row in ordered] == ["d", "c", "a", "b"]
            bounds = b.execute_query('SELECT MIN("price") AS low, MAX("price") AS high FROM "rates"')[0]
            assert b.deserialize_field(bounds["low"], code) == Decimal("-10.00")
            assert b.deserialize_field(bounds["high"], code) == Decimal("10.00")
            # an operand is compared through numeric_comparison, against the scaled column
            condition, params = b.numeric_comparison('"price"', ">", Decimal("9.6"), code)
            above = b.execute_query(f'SELECT "id" FROM "rates" WHERE {condition} ORDER BY "id"', params)
            assert [row["id"] for row in above] == ["b"]
            total = b.execute_query('SELECT SUM("price") AS total FROM "rates"')[0]["total"]
            assert b.deserialize_field(total, code) == Decimal("7.25")
            assert str(b.deserialize_field(total, code)) == "7.25"
        finally:
            b.reset()

    _STORED: ClassVar[dict[str, Decimal | None]] = {
        "a": Decimal("10.00"),
        "b": Decimal("-10.00"),
        "c": Decimal("9.99"),
        "d": Decimal("-9.99"),
        "e": None,
    }
    _OPERATORS: ClassVar[dict[str, Callable[[Decimal, Decimal], bool]]] = {
        "=": operator.eq,
        "!=": operator.ne,
        "<>": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    def _matching(self, b: SQLiteBackend, op: str, operand: Any) -> list[str]:
        for key, price in self._STORED.items():
            b.upsert("rates", {"id": key, "price": price})
        condition, params = b.numeric_comparison('"price"', op, operand, b.column_types("rates")["price"])
        rows = b.execute_query(f'SELECT "id" FROM "rates" WHERE {condition} ORDER BY "id"', params)
        return [row["id"] for row in rows]

    @pytest.mark.parametrize("op", ["=", "!=", "<>", "<", "<=", ">", ">="])
    @pytest.mark.parametrize(
        "operand",
        [
            Decimal("9.995"),
            Decimal("-9.995"),
            "9.995",
            9.995,
            Decimal("10.00"),
            Decimal("10"),
            Decimal("-10.000"),
            10,
            Decimal("9.99"),
            Decimal("-9.99"),
            Decimal("1E+30"),
            Decimal("-1E+30"),
            Decimal("0.0000001"),
        ],
    )
    def test_an_operand_finer_than_the_scale_compares_as_l3_compares_it(self, op: str, operand: Any) -> None:
        """``price > 9.995`` against ``10.00`` is true in L3; rounded to the scale first, it was ``1000 > 1000``."""
        b = self._backend()
        try:
            exact = Decimal(repr(operand)) if isinstance(operand, float) else Decimal(operand)
            expected = sorted(
                key for key, price in self._STORED.items() if price is not None and self._OPERATORS[op](price, exact)
            )
            assert self._matching(b, op, operand) == expected
        finally:
            b.reset()

    @pytest.mark.parametrize(
        ("operand", "above"),
        [(Decimal("NaN"), True), (Decimal("Infinity"), True), (float("inf"), True), (Decimal("-Infinity"), False)],
    )
    @pytest.mark.parametrize("op", ["=", "!=", "<>", "<", "<=", ">", ">="])
    def test_a_non_finite_operand_compares_above_or_below_every_value(self, op: str, operand: Any, above: bool) -> None:
        b = self._backend()
        try:
            held = sorted(key for key, price in self._STORED.items() if price is not None)
            true_for = {"!=", "<>", "<", "<="} if above else {"!=", "<>", ">", ">="}
            assert self._matching(b, op, operand) == (held if op in true_for else [])
        finally:
            b.reset()

    def test_an_unknown_operator_or_a_column_that_is_not_numeric_is_refused(self) -> None:
        b = self._backend()
        try:
            with pytest.raises(ValueError, match="LIKE"):
                b.numeric_comparison('"price"', "LIKE", 1, b.column_types("rates")["price"])
            with pytest.raises(ValueError, match="TEXT"):
                b.numeric_comparison('"id"', "=", 1, b.column_types("rates")["id"])
        finally:
            b.reset()

    def test_a_sum_past_a_64_bit_integer_raises_naming_the_column(self) -> None:
        """ten of NUMERIC(18, 2)'s largest values total past 2**63 once scaled; SQLite says only "integer overflow"."""
        metadata = MetaData()
        TableSchema(
            name="ledger",
            primary_key="id",
            columns=[SchemaColumn("id", STRING_TYPE), SchemaColumn("amount", NUMERIC_TYPE, precision=18, scale=2)],
        ).to_sqlalchemy_table(metadata)
        b = SQLiteBackend(db_name=f"test_numeric_sum_{uuid.uuid4().hex[:8]}")
        b.initialize(metadata)
        try:
            b.upsert_many("ledger", [{"id": str(i), "amount": Decimal("9999999999999999.99")} for i in range(10)], "id")
            with pytest.raises(OverflowError, match=r"ledger\.amount"):
                b.execute_query('SELECT SUM("amount") AS total FROM "ledger"')
            # a statement that fails some other way raises as SQLite raised it
            with pytest.raises(sqlite3.OperationalError, match="no such column"):
                b.execute_query('SELECT SUM(missing) FROM "ledger"')
        finally:
            b.reset()

    @pytest.mark.parametrize(
        ("sa_type", "reason"),
        [(Numeric(), "no declared precision and scale"), (Numeric(19, 2), r"NUMERIC\(19, 2\)")],
    )
    def test_a_numeric_this_l1_cannot_hold_exactly_is_refused_naming_its_column(
        self, sa_type: Numeric, reason: str
    ) -> None:
        metadata = MetaData()
        Table("wide", metadata, Column("id", String(255), primary_key=True), Column("amount", sa_type))
        b = SQLiteBackend(db_name=f"test_numeric_refused_{uuid.uuid4().hex[:8]}")
        try:
            with pytest.raises(UnsupportedL1ColumnError, match=rf"wide\.amount: .*{reason}"):
                b.initialize(metadata)
            assert not b.has_table("wide")
        finally:
            b.reset()
