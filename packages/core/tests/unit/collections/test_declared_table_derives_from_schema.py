"""a declared collection reads its row types and write columns off its schema, and refuses a copy that disagrees.

A :class:`DeclaredTableCollection` used to need a hand-written field-type map for its L2 codec and
a hand-written column list for its upsert, beside the ``TableSchema`` that already said both, and
nothing held the copies to the schema. Pinned here:

- a collection that writes neither hands back every type (``Decimal``, an aware instant, a UUID,
  JSON, a bool, bytes, integers) from its SQLite L1, DuckDB L1 and L2 codec;
- its upsert names the columns the row carries, rewrites the mutable ones only, and conflicts on
  its key or the natural key it names;
- a class that still states ``datetime_columns``, ``field_types``, ``insert_columns`` or
  ``update_columns`` is refused with a ``TypeError`` when it disagrees with the schema, and
  accepted when it agrees; a ``SchemaBackedCollection`` is held to the same rule.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
import dataclasses
from decimal import Decimal
from typing import Any, ClassVar, cast

import pytest

from threetears.core.collections import DeclaredTableCollection
from threetears.core.collections.base import NO_L2
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import ROW_STAMPED, AppendOnlyWriteError, next_epoch, row_versions
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.tier_parity import tier_parity_findings
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_SCHEMA = TableSchema(
    name="every_type_rows",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("code", STRING_TYPE),
        Column("price", NUMERIC_TYPE, precision=12, scale=4),
        Column("owner_id", UUID_TYPE, nullable=True),
        Column("doc", JSONB_TYPE, nullable=True),
        Column("active", BOOL_TYPE, server_default="false"),
        Column("blob", BYTES_TYPE, nullable=True),
        Column("small", INT_TYPE, nullable=True),
        Column("big", BIGINT_TYPE, nullable=True),
        Column("seen_at", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    l1_eviction=TEST_L1_TABLE,
)

_TYPES: dict[str, Any] = {
    "id": uuid.UUID,
    "code": str,
    "price": Decimal,
    "owner_id": uuid.UUID,
    "doc": dict,
    "active": bool,
    "blob": bytes,
    "small": int,
    "big": int,
    "seen_at": datetime | None,
    "date_created": datetime,
    "date_updated": datetime,
}


class _Row(BaseEntity):
    primary_key_field = "id"


class _Every(DeclaredTableCollection[_Row]):
    """a collection of every column type that writes no field-type map and no column list."""

    schema = _SCHEMA

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        return None


def _every() -> _Every:
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return _Every(registry=CollectionRegistry(), config=config, nats_client=NO_L2)


def _row() -> dict[str, Any]:
    now = datetime(2026, 10, 24, 12, 34, 56, 789012, tzinfo=UTC)
    return {
        "id": uuid.uuid4(),
        "code": "a",
        "price": Decimal("-12345678.1234"),
        "owner_id": uuid.uuid4(),
        "doc": {"nested": {"n": 1, "l": [1, "two", None]}, "b": True},
        "active": True,
        "blob": b"\x00\xffbytes",
        "small": -2147483647,
        "big": 9007199254740993,
        "seen_at": now,
        "date_created": now,
        "date_updated": now,
    }


class TestEveryTierHandsBackEveryType:
    """with no hand-written map, the L1s and the L2 codec hand back what was written."""

    def test_the_l1s_and_the_l2_codec_hand_back_every_type(self) -> None:
        assert tier_parity_findings([_Every]) == []

    def test_the_l2_codec_rehydrates_each_column_to_its_declared_type(self) -> None:
        row = _row()
        decoded = _every().decode_row(_every().serialize(row))
        assert decoded == row
        for name, value in row.items():
            assert type(decoded[name]) is type(value), name
        assert decoded["seen_at"].tzinfo is not None

    def test_the_row_types_are_read_off_the_schema(self) -> None:
        assert _Every.datetime_columns == {"seen_at", "date_created", "date_updated"}
        assert _Every.field_types["price"] is Decimal
        assert _Every.field_types["blob"] is bytes
        assert list(_Every.field_types) == [column.name for column in _SCHEMA.columns]


class TestTheWriteColumnsAreReadOffTheSchema:
    """the upsert names the carried columns and rewrites only the mutable ones."""

    def test_the_insert_and_update_columns(self) -> None:
        assert _Every.insert_columns == tuple(column.name for column in _SCHEMA.columns)
        # not the key, and not the immutable ``date_created``
        assert "id" not in _Every.update_columns
        assert "date_created" not in _Every.update_columns
        assert _Every.conflict_columns == ("id",)

    def test_a_full_row_is_an_upsert_on_the_key_binding_each_value_by_its_type(self) -> None:
        row = _row()
        sql, params = _every().write_statement(row)
        assert sql.startswith(f"INSERT INTO every_type_rows ({', '.join(_Every.insert_columns)}) VALUES ($1, $2")
        assert "$5::jsonb" in sql
        assert "ON CONFLICT (id) DO UPDATE SET code = EXCLUDED.code" in sql
        assert "date_created = EXCLUDED" not in sql
        assert params[_Every.insert_columns.index("price")] == row["price"]
        assert params[_Every.insert_columns.index("doc")] == row["doc"]

    def test_a_column_the_row_does_not_carry_is_not_written(self) -> None:
        row = _row()
        del row["blob"]
        row["active"] = None  # NOT NULL with a default: not a value the column can hold
        sql, params = _every().write_statement(row)
        assert "blob" not in sql and "active" not in sql
        assert len(params) == len(_Every.insert_columns) - 2

    def test_a_natural_key_conflicts_on_it_and_returns_the_stored_key(self) -> None:
        class _ByCode(_Every):
            conflict_columns: ClassVar[tuple[str, ...]] = ("code",)

        assert _ByCode.update_columns == tuple(c for c in _Every.update_columns if c != "code")
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        sql, _ = _ByCode(registry=CollectionRegistry(), config=config, nats_client=NO_L2).write_statement(
            _row(), returning=True
        )
        assert "ON CONFLICT (code) DO UPDATE SET price = EXCLUDED.price" in sql
        assert sql.endswith("RETURNING id")


class TestACopyThatDisagreesIsRefused:
    """a hand-written copy beside the schema must say what the schema says."""

    def test_an_agreeing_field_type_map_and_instants_are_accepted(self) -> None:
        class _Agreeing(_Every):
            schema = _SCHEMA
            field_types: ClassVar[dict[str, Any]] = _TYPES
            datetime_columns: ClassVar[frozenset[str]] = frozenset({"seen_at", "date_created", "date_updated"})

        assert _Agreeing.field_types["price"] is Decimal
        # restating agreeing copies leaves the class every other derivation
        assert _Agreeing.insert_columns == _Every.insert_columns
        assert _Agreeing.update_columns == _Every.update_columns

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"price": float}, "'price' is declared <class 'float'>"),
            ({"blob": str}, "'blob' is declared <class 'str'>"),
            ({"extra": str}, "'extra' is not a column of 'every_type_rows'"),
        ],
    )
    def test_a_field_type_map_that_disagrees_is_refused(self, change: dict[str, Any], message: str) -> None:
        with pytest.raises(TypeError, match=message):

            class _Drifted(_Every):
                field_types: ClassVar[dict[str, Any]] = {**_TYPES, **change}

    def test_a_field_type_map_missing_a_column_is_refused(self) -> None:
        short = {name: hint for name, hint in _TYPES.items() if name != "seen_at"}
        with pytest.raises(TypeError, match="'seen_at' is missing"):

            class _Short(_Every):
                field_types: ClassVar[dict[str, Any]] = short

    def test_instants_that_disagree_are_refused(self) -> None:
        with pytest.raises(TypeError, match="declares datetime_columns"):

            class _Drifted(_Every):
                datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def test_an_insert_column_list_that_disagrees_is_refused(self) -> None:
        with pytest.raises(TypeError, match="declares insert_columns"):

            class _Drifted(_Every):
                insert_columns: ClassVar[tuple[str, ...]] = ("id", "code", "price")

    def test_an_update_column_list_that_disagrees_is_refused(self) -> None:
        with pytest.raises(TypeError, match="declares update_columns"):

            class _Drifted(_Every):
                update_columns: ClassVar[tuple[str, ...]] = (*_Every.update_columns, "date_created")

    def test_an_agreeing_column_list_is_accepted(self) -> None:
        class _Agreeing(_Every):
            insert_columns: ClassVar[tuple[str, ...]] = _Every.insert_columns

        assert _Agreeing.insert_columns == _Every.insert_columns

    def test_a_conflict_target_the_schema_lacks_is_refused(self) -> None:
        with pytest.raises(TypeError, match=r"\['sku'\] are not columns"):

            class _Drifted(_Every):
                conflict_columns: ClassVar[tuple[str, ...]] = ("sku",)

    def test_a_schema_backed_collection_is_held_to_the_same_rule(self) -> None:
        with pytest.raises(TypeError, match="'price' is declared <class 'float'>"):

            class _Generated(SchemaBackedCollection[_Row]):
                schema = _SCHEMA
                field_types: ClassVar[dict[str, Any]] = {**_TYPES, "price": float}

        with pytest.raises(TypeError, match="declares datetime_columns"):

            class _GeneratedInstants(SchemaBackedCollection[_Row]):
                schema = _SCHEMA
                datetime_columns: ClassVar[frozenset[str]] = frozenset({"seen_at"})


_STAMPED_SCHEMA = dataclasses.replace(_SCHEMA, name="stamped_rows", row_epoch=ROW_STAMPED)


class _Stamped(_Every):
    """the same table declared ROW_STAMPED: one row a key, updated in place, its epoch rising on every write."""

    schema = _STAMPED_SCHEMA


class TestAStampedTablesOwnStatementsStampItsRows:
    """a collection whose statements are its own stamps a stamped table's rows as the schema-backed writes do."""

    async def test_an_upsert_and_an_update_carry_a_fresh_stamp_that_never_goes_back(self) -> None:
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        collection = _Stamped(registry=CollectionRegistry(), config=config, nats_client=NO_L2)
        sent: list[tuple[str, tuple[Any, ...]]] = []

        class _Pool:
            async def execute(self, sql: str, *params: Any) -> str:
                sent.append((sql, params))
                return "INSERT 0 1"

        collection.l3_pool = cast(Any, _Pool())
        before = next_epoch()
        await collection.write_row(_row())
        partial = {"id": uuid.uuid4(), "code": "b"}
        await collection.write_row(partial)
        (upsert, upsert_params), (update, update_params) = sent
        assert "epoch = GREATEST(stamped_rows.epoch + 1, EXCLUDED.epoch)" in upsert
        assert "epoch = GREATEST(epoch + 1, $" in update
        for params in (upsert_params, update_params):
            [stamp] = [value for value in params if isinstance(value, int) and value > before]
            assert stamp > before

    def test_a_natural_key_upsert_arbitrates_on_live_rows_only(self) -> None:
        """a stamped table's natural name holds its live rows only (``UNIQUE (...) WHERE NOT gone``), so an upsert on
        it names that predicate (PostgreSQL infers the partial index only from it), and an update by it reaches a live
        row only; the primary key keeps its plain target (a write on it revives a gone row in place)."""

        class _StampedByCode(_Stamped):
            conflict_columns: ClassVar[tuple[str, ...]] = ("code",)

        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        by_code = _StampedByCode(registry=CollectionRegistry(), config=config, nats_client=NO_L2)
        stamped = {**_row(), "epoch": 1, "gone": False}
        upsert, _ = by_code.write_statement(stamped, returning=True)
        assert "ON CONFLICT (code) WHERE NOT gone DO UPDATE SET" in upsert
        update, _ = by_code.write_statement({"id": uuid.uuid4(), "code": "b", "price": 1, "epoch": 1, "gone": False})
        assert update.startswith("UPDATE stamped_rows SET") and update.endswith("AND NOT gone")
        by_key = _Stamped(registry=CollectionRegistry(), config=config, nats_client=NO_L2)
        keyed, _ = by_key.write_statement({**_row(), "epoch": 1, "gone": False})
        assert "ON CONFLICT (id) DO UPDATE SET" in keyed and "WHERE NOT gone" not in keyed

    async def test_a_versioned_tables_upsert_is_refused(self) -> None:
        class _Versioned(_Every):
            schema = dataclasses.replace(
                _SCHEMA,
                name="versioned_rows",
                row_epoch=row_versions("code"),
            )

        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        collection = _Versioned(registry=CollectionRegistry(), config=config, nats_client=NO_L2)
        collection.l3_pool = cast(Any, object())
        with pytest.raises(AppendOnlyWriteError, match="versioned"):
            await collection.write_row(_row())
