"""the order columns on a schema-backed table, the SQL that fences on them, and the order helpers.

What this pins:

- a table declares both order columns or neither, of the right types and mutable, on an
  ``on_conflict="update"`` table without ``cas_null_safe`` -- refused at class definition otherwise;
- the generated upsert lands only over a strictly older stored order, a ``NULL`` one included;
- a schema-backed collection persists the order exactly when its table declares it and its store
  can write fenced on it, and an ordered write to anything else is refused;
- the order helpers read the epoch whichever way a tier hands it back, and the migration helper
  refuses to interpolate anything but a table name.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from threetears.core.backends import schema_sql
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.l2_order import (
    L2_ORDER_FLOOR,
    L2Order,
    l2_order_migration_statements,
    l2_order_of,
    with_l2_order,
    without_l2_order,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    l2_order_columns,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

_EPOCH = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _schema(*extra: Column, **kwargs: Any) -> TableSchema:
    return TableSchema(
        name="member_sets",
        primary_key="id",
        columns=[
            Column("id", STRING_TYPE),
            Column("members", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, nullable=True, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
            *extra,
        ],
        **kwargs,
    )


class TestTheSchemaDeclaresBothColumnsOrNeither:
    def test_both_columns_from_the_helper_are_accepted(self) -> None:
        assert _schema(*l2_order_columns()).declares_l2_order

    def test_a_table_without_them_declares_no_order(self) -> None:
        assert not _schema().declares_l2_order

    def test_one_column_alone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            _schema(Column("l2_epoch", DATETIMETZ_TYPE, nullable=True))

    @pytest.mark.parametrize(
        "revision",
        [
            Column("l2_revision", INT_TYPE, nullable=True),  # 32 bits: a busy bucket passes it
            Column("l2_revision", BIGINT_TYPE),  # not nullable: a row another path wrote has none
            Column("l2_revision", BIGINT_TYPE, nullable=True, immutable=True),  # the upsert must rewrite it
        ],
    )
    def test_an_unsound_revision_column_is_refused(self, revision: Column) -> None:
        with pytest.raises(ValueError, match="l2_revision"):
            _schema(Column("l2_epoch", DATETIMETZ_TYPE, nullable=True), revision)

    def test_an_epoch_of_the_wrong_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="l2_epoch"):
            _schema(Column("l2_epoch", STRING_TYPE, nullable=True), Column("l2_revision", BIGINT_TYPE, nullable=True))

    def test_a_table_that_ignores_conflicts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="on_conflict='update'"):
            _schema(*l2_order_columns(), on_conflict="ignore")


class TestTheOrderedUpsert:
    def test_it_updates_only_over_a_strictly_older_stored_order(self) -> None:
        schema = _schema(*l2_order_columns())
        row = with_l2_order({"id": "s", "members": "[]"}, L2Order(_EPOCH, 7))
        sql = schema_sql.build_ordered_upsert_sql(schema, row)
        assert sql == (
            "INSERT INTO member_sets (id, members, date_created, date_updated, l2_epoch, l2_revision) "
            "VALUES ($1, $2, $3, $4, $5, $6) "
            "ON CONFLICT (id) DO UPDATE SET members = EXCLUDED.members, date_updated = EXCLUDED.date_updated, "
            "l2_epoch = EXCLUDED.l2_epoch, l2_revision = EXCLUDED.l2_revision "
            "WHERE (COALESCE(member_sets.l2_epoch, '-infinity'::timestamptz), COALESCE(member_sets.l2_revision, -1)) "
            "< (EXCLUDED.l2_epoch, EXCLUDED.l2_revision)"
        )
        params = schema_sql.build_insert_params(schema, row)
        assert params[4:] == [_EPOCH, 7]

    def test_a_row_without_its_order_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="l2_epoch"):
            schema_sql.build_ordered_upsert_sql(_schema(*l2_order_columns()), {"id": "s"})

    def test_a_table_without_the_columns_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="mutable column"):
            schema_sql.build_ordered_upsert_sql(_schema(), with_l2_order({"id": "s"}, L2Order(_EPOCH, 1)))


class TestTheOrderHelpers:
    def test_an_order_is_read_from_a_datetime_or_its_iso_string(self) -> None:
        order = L2Order(_EPOCH, 3)
        assert l2_order_of(with_l2_order({}, order)) == order
        assert l2_order_of({"l2_epoch": _EPOCH.isoformat(), "l2_revision": "3"}) == order

    def test_a_row_missing_either_half_carries_no_order(self) -> None:
        assert l2_order_of({"l2_epoch": _EPOCH}) is None
        assert l2_order_of(without_l2_order(with_l2_order({}, L2Order(_EPOCH, 3)))) is None

    def test_a_later_incarnation_orders_after_every_revision_of_an_earlier_one(self) -> None:
        assert L2Order(_EPOCH.replace(second=1), 1) > L2Order(_EPOCH, 10**12)
        assert L2_ORDER_FLOOR < L2Order(_EPOCH, 0)

    def test_a_naive_epoch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            L2Order(datetime(2026, 9, 26), 1)  # noqa: DTZ001 - the naive value is the point

    def test_the_migration_refuses_anything_but_a_table_name(self) -> None:
        assert len(l2_order_migration_statements("agent_ab.indexes")) == 3
        with pytest.raises(ValueError, match="plain SQL table name"):
            l2_order_migration_statements("indexes; DROP TABLE users")


class _Row(BaseEntity):
    primary_key_field = "id"


class _Sets(SchemaBackedCollection[_Row]):
    schema: ClassVar[TableSchema] = _schema(*l2_order_columns())  # type: ignore[misc]

    @property
    def table_name(self) -> str:
        return "member_sets"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _UnorderedStore:
    """a durable store with the structured seam but no ordered write -- a git backend's shape."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        return self.rows.get(pk["id"])

    async def upsert(self, table: str, row: dict[str, Any], **kwargs: Any) -> int:
        self.rows[row["id"]] = dict(row)
        return 1

    async def delete(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> None:
        self.rows.pop(pk["id"], None)

    async def scan(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self.rows.values())


def _sets(store: Any, nats: FakeNatsClient) -> _Sets:
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=SQLiteBackend(db_name=f"sets_{uuid.uuid4().hex[:8]}"),
        l2_client=nats,
        l3_pool=store,
        kv_key_scope="sets-principal",
    )
    return _Sets(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


class TestTheCollectionPersistsTheOrderOnlyWhereItCan:
    @pytest.mark.asyncio
    async def test_a_store_that_cannot_write_ordered_is_refused_before_l2(self) -> None:
        nats, store = FakeNatsClient(), _UnorderedStore()
        sets = _sets(store, nats)
        assert not sets.persists_l2_order
        with pytest.raises(ValueError, match="l2_epoch"):
            await sets.l2_cas_mutate("s", lambda _row: ("upsert", {"id": "s", "members": "[]"}))
        assert (await nats.kv_bucket(name="collections")).keys() == ()
        with pytest.raises(TypeError, match="OrderedDurableStore"):
            await sets.save_ordered_to_store(with_l2_order({"id": "s"}, L2Order(_EPOCH, 1)))
