"""the row epoch (``design-append-only-l3.md``, chunk 1): its encoding, a table's declaration of it and the DDL that
follows, and the refusal of a write that carries none.

The epoch is ``version << 31 | sequence``: every later draw sorts newer, within one version and across versions. A
versioned table's stored key is its key and then the epoch, newest first on YugabyteDB; a stamped table keeps its key and carries the column. Every write path the store owns refuses a row of such a
table without an epoch, before anything is sent.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
from sqlalchemy import MetaData
import pytest

from threetears.core.backends.protocol import ArrowWritePlan, LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    Index,
    TableSchema,
    UniqueConstraintDef,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import (
    EPOCH_COLUMN,
    ROW_STAMPED,
    RowEpoch,
    UnstampedWriteError,
    next_epoch,
    row_versions,
)
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE


def _versioned(name: str = "results", **extra: Any) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key=("office_key", "candidate"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("candidate", STRING_TYPE),
            Column("join_state_code", STRING_TYPE),
            Column("votes", INT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        row_epoch=row_versions("join_state_code"),
        l1_eviction=TEST_L1_TABLE,
        **extra,
    )


def _stamped(name: str = "runs") -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="run_id",
        columns=[Column("run_id", STRING_TYPE), Column("state", STRING_TYPE, nullable=True)],
        row_epoch=ROW_STAMPED,
        l1_eviction=TEST_L1_TABLE,
    )


# -- the encoding ----------------------------------------------------------------------------------------------------


def _version(epoch: int) -> int:
    """the version an epoch was stamped at, by its documented encoding (``version << 31 | sequence``)."""
    return epoch >> 31


def test_an_epoch_carries_its_writes_version_in_its_high_bits_and_fits_a_bigint() -> None:
    epoch = next_epoch(1234)
    assert _version(epoch) == 1234
    assert 0 < epoch & ((1 << 31) - 1), "the sequence is drawn from 1"
    # every epoch of a version sorts below every epoch of the next, whatever the sequence
    assert next_epoch(7) < (8 << 31) <= next_epoch(8)
    # the largest version still fits a BIGINT
    assert next_epoch((1 << 32) - 1) < 1 << 63


@pytest.mark.parametrize("version", [-1, 1 << 32])
def test_a_version_out_of_range_is_refused_never_wrapped(version: int) -> None:
    with pytest.raises(ValueError, match="must be within"):
        next_epoch(version)


def test_each_draw_of_a_version_sorts_newer_than_every_draw_before_it_from_any_thread() -> None:
    drawn: list[list[int]] = [[] for _ in range(4)]

    def draw(into: list[int]) -> None:
        into.extend(next_epoch(9) for _ in range(2000))

    threads = [threading.Thread(target=draw, args=(each,)) for each in drawn]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    every = [epoch for each in drawn for epoch in each]
    assert len(set(every)) == len(every), "two write statements drew one epoch"
    assert all(each == sorted(each) for each in drawn), "a later draw on one thread sorted older"
    assert {_version(epoch) for epoch in every} == {9}
    assert next_epoch(9) > max(every)


def test_a_versioned_declaration_names_its_scope_and_a_stamped_one_none() -> None:
    with pytest.raises(ValueError, match="scope"):
        RowEpoch(versions=True)
    with pytest.raises(ValueError, match="no scope"):
        RowEpoch(versions=False, scope="join_state_code")


# -- the declaration and its DDL --------------------------------------------------------------------------------------


def test_a_versioned_table_is_stored_by_its_key_then_the_epoch() -> None:
    schema = _versioned()
    assert schema.pk_columns == ("office_key", "candidate")
    assert schema.stored_key == ("office_key", "candidate", EPOCH_COLUMN)
    epoch = schema.column(EPOCH_COLUMN)
    assert (epoch.column_type, epoch.nullable, epoch.immutable, epoch.server_default) == (
        BIGINT_TYPE,
        False,
        True,
        None,
    )
    # its (scope, epoch) index is declared with the restore read that names it (append-only chunk 5)
    assert schema.indexes == ()
    # never in an upsert's SET: it is part of the stored key
    assert EPOCH_COLUMN not in [column.name for column in schema.mutable_columns()]


def test_a_stamped_table_keeps_its_key_and_sets_the_epoch_on_each_write() -> None:
    schema = _stamped()
    assert schema.stored_key == schema.pk_columns == ("run_id",)
    assert EPOCH_COLUMN in [column.name for column in schema.mutable_columns()]
    assert schema.indexes == ()


def test_a_table_that_declares_no_epoch_is_as_before() -> None:
    schema = TableSchema(name="plain", primary_key="k", columns=[Column("k", STRING_TYPE)], l1_eviction=TEST_L1_TABLE)
    assert schema.stored_key == ("k",)
    assert not schema.stamps_epoch
    assert schema.get_column(EPOCH_COLUMN) is None


def test_a_versioned_table_on_yugabyte_hashes_its_first_key_column_and_orders_the_epoch_newest_first() -> None:
    [create] = _versioned().create_statements(yugabyte=True)
    assert "epoch BIGINT NOT NULL" in create
    assert "PRIMARY KEY ((office_key) HASH, candidate ASC, epoch DESC)" in create


def test_a_versioned_table_on_postgres_keeps_the_epoch_last_in_its_primary_key() -> None:
    [create] = _versioned().create_statements(yugabyte=False)
    assert "PRIMARY KEY (office_key, candidate, epoch)" in create


def test_the_epoch_is_last_in_the_stored_key_whatever_the_columns_order() -> None:
    schema = TableSchema(
        name="ordered",
        primary_key=("b", "a"),
        columns=[Column("a", STRING_TYPE), Column("b", STRING_TYPE), Column("s", STRING_TYPE)],
        row_epoch=row_versions("s"),
        l1_eviction=TEST_L1_TABLE,
    )
    [create] = schema.create_statements(yugabyte=True)
    assert "PRIMARY KEY ((b) HASH, a ASC, epoch DESC)" in create


def test_a_stamped_table_carries_the_column_and_no_other_change() -> None:
    [create] = _stamped().create_statements(yugabyte=True)
    assert "epoch BIGINT NOT NULL" in create
    assert "PRIMARY KEY (run_id)" in create


@pytest.mark.parametrize(
    ("extra", "what"),
    [
        ({"unique_constraints": (UniqueConstraintDef("results_one", ("office_key",)),)}, "unique constraint"),
        ({"indexes": (Index("results_u", "office_key", unique=True),)}, "unique index"),
        ({"cas_column": "date_updated"}, "cas_column"),
    ],
)
def test_a_versioned_table_refuses_what_would_fence_one_version_against_another(
    extra: dict[str, Any], what: str
) -> None:
    with pytest.raises(ValueError, match=what):
        _versioned(**extra)


def test_an_epoch_column_declared_otherwise_or_named_in_the_key_is_refused() -> None:
    with pytest.raises(ValueError, match="BIGINT NOT NULL"):
        TableSchema(
            name="t",
            primary_key="k",
            columns=[Column("k", STRING_TYPE), Column(EPOCH_COLUMN, BIGINT_TYPE, nullable=True)],
            row_epoch=ROW_STAMPED,
            l1_eviction=TEST_L1_TABLE,
        )
    with pytest.raises(ValueError, match="not named in primary_key"):
        TableSchema(
            name="t",
            primary_key=("k", EPOCH_COLUMN),
            columns=[Column("k", STRING_TYPE), Column(EPOCH_COLUMN, BIGINT_TYPE)],
            row_epoch=row_versions("k"),
            l1_eviction=TEST_L1_TABLE,
        )
    with pytest.raises(ValueError, match="scope"):
        TableSchema(
            name="t",
            primary_key="k",
            columns=[Column("k", STRING_TYPE)],
            row_epoch=row_versions("no_such_column"),
            l1_eviction=TEST_L1_TABLE,
        )


def test_an_upsert_of_a_versioned_table_is_planned_as_an_append_with_no_conflict_clause() -> None:
    # append-only L3, chunk 2: chunk 1's interim upsert on (key..., epoch) is gone; a rewrite appends a new version
    store = SqlL3Backend(MagicMock())
    store.register_schema("results", _versioned())
    plan = store.arrow_write_plan(
        "results",
        ["office_key", "candidate", "join_state_code", "date_created", "date_updated", EPOCH_COLUMN],
        [pa.string(), pa.string(), pa.string(), pa.timestamp("us", tz="UTC"), pa.timestamp("us", tz="UTC"), pa.int64()],
        kind="upsert",
        conn=MagicMock(write_arrow=True),
    )
    assert plan is not None
    assert plan.kind == "append"
    assert (plan.conflict_columns, plan.update_columns) == ((), ())


# -- a write that carries no epoch is refused -------------------------------------------------------------------------


class _Row(BaseEntity):
    primary_key_field = "candidate"


def _collection(schema: TableSchema) -> Any:
    store = SqlL3Backend(MagicMock())
    registry = CollectionRegistry()
    registry.configure(l3_pool=store)
    return collection_for_schema(schema, entity_class=_Row)(registry, DefaultCoreConfig(), None)


async def test_a_transaction_that_took_no_writes_turn_has_no_epoch_to_give() -> None:
    conn = MagicMock()
    transaction = CallerTransaction(conn)
    with pytest.raises(UnstampedWriteError, match="took no write's turn"):
        transaction.row_epoch("results")
    transaction.note_write_version(41)
    first, second = transaction.row_epoch("results"), transaction.row_epoch("results")
    assert _version(first) == _version(second) == 41
    assert second > first


async def test_a_collection_write_on_no_transaction_or_one_that_took_no_turn_is_refused() -> None:
    collection = _collection(_versioned())
    row = {"office_key": "sn_NC1", "candidate": "a", "join_state_code": "NC"}
    with pytest.raises(UnstampedWriteError, match="joins no transaction"):
        await collection.save_to_store(row)


async def test_every_store_write_of_a_row_with_no_epoch_is_refused_before_anything_is_sent() -> None:
    pool = MagicMock()
    store = SqlL3Backend(pool)
    store.register_schema("results", _versioned())
    store.register_schema("runs", _stamped())
    row = {"office_key": "sn_NC1", "candidate": "a", "join_state_code": "NC", "votes": 1}
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.upsert("results", row)
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.upsert_many("results", [row], max_rows=10, max_bytes=1 << 20)
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.copy_many("results", [row], max_rows=10, max_bytes=1 << 20, conn=MagicMock())
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.upsert_many("runs", [{"run_id": "r1", EPOCH_COLUMN: None}], max_rows=10, max_bytes=1 << 20)
    rows = pa.table({"office_key": ["sn_NC1"], "candidate": ["a"], "join_state_code": ["NC"]})
    plan = ArrowWritePlan(kind="upsert", columns=(), conflict_columns=("office_key", "candidate", EPOCH_COLUMN))
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.write_arrow("results", rows, plan, conn=MagicMock())
    nulls = rows.append_column(EPOCH_COLUMN, pa.array([None], pa.int64()))
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.write_arrow("results", nulls, plan, conn=MagicMock())
    with pytest.raises(UnstampedWriteError, match="has none"):
        await store.load_files(
            "results",
            [LoadFile(url="file:///nowhere.parquet", rows=1)],
            columns=["candidate", "join_state_code"],
            constants={"office_key": "sn_NC1"},
            rows_per_transaction=10,
            timeout_ms=1000,
        )
    pool.execute.assert_not_called()
    pool.acquire.assert_not_called()


async def test_a_load_of_a_table_that_declares_an_epoch_names_its_writes_version() -> None:
    collection = _collection(_versioned())
    with pytest.raises(UnstampedWriteError, match="write_version"):
        await collection.load_files(
            [LoadFile(url="file:///nowhere.parquet", rows=1)],
            columns=["candidate", "join_state_code"],
            constants={"office_key": "sn_NC1"},
        )


def test_a_versioned_tables_cache_holds_one_row_a_key() -> None:
    """a process's L1 is made from ``to_sqlalchemy_table``: one row a key, the newest, never a row a version."""
    table = _versioned().to_sqlalchemy_table(MetaData())
    assert [column.name for column in table.primary_key.columns] == ["office_key", "candidate"]
    assert EPOCH_COLUMN in table.columns
