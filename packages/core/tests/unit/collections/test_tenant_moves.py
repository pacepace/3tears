"""A stamped row that moves between tenants records the tenant it left on itself, and both tenants hear of it.

Pace's option A (2026-11-25, ``threetears.core.ddl.tenant_moves``): with the change-note ring retired, nothing but the
row can tell the tenant a write moved it off that the row left. A ``moveable`` tenancy column carries the L3-only
``<column>_moved_from`` and ``<column>_moved_epoch``, which the write that changes the column fills in the same
statement; the write's epoch record names the old tenant beside the new one, and a restore of a lost log reads each
moved row back for the tenant it left. Each test here flips on the code before it: no column, no SET term, one record,
one tenant restored.

The pools are stand-ins recording every statement; the statements' meaning on a real database (a move fills both
columns, a write that does not move leaves them) is pinned by the SQL each one renders.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from threetears.core.backends import schema_sql
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.generation import WRITE_GENERATION
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
    stamped_values,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, ROW_STAMPED
from threetears.core.ddl.tenant_moves import moved_epoch_column, moved_from_column, taken_moves
from threetears.core.entities.base import BaseEntity
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import epoch_change_key, tenant_column, value_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats import set_default_namespace

_TABLE = "moving_widgets"
_TENANT = "customer_id"
_MOVED_FROM = moved_from_column(_TENANT)
_MOVED_EPOCH = moved_epoch_column(_TENANT)

#: a platform table whose tenant column a write may change in place: promotion candidates' shape
_WIDGETS = TableSchema(
    name=_TABLE,
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column(_TENANT, UUID_TYPE, nullable=True),
        Column("label", STRING_TYPE),
        Column("rows", BIGINT_TYPE),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    tenancy=tenant_column(_TENANT),
    row_epoch=ROW_STAMPED,
    write_generation=WRITE_GENERATION,
    moveable=(_TENANT,),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table whose tenant moves"),
)

_OLD = "0193a2b4-0000-7000-8000-00000000000a"
_NEW = "0193a2b4-0000-7000-8000-00000000000b"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("tenantmoves")


def _declaration(schema: TableSchema) -> str:
    """the restore declaration the table's DDL gives its restore index, as the catalog then holds it."""
    [comment] = [statement for statement in schema.create_statements(yugabyte=True) if statement.startswith("COMMENT")]
    found = re.search(r"IS '(.*)'$", comment)
    assert found is not None, comment
    return found.group(1).replace("''", "'")


class _Widget(BaseEntity):
    primary_key_field = "id"


def test_a_moveable_column_carries_its_l3_only_columns_and_its_restore_declaration_names_it() -> None:
    """Flips if the L3 table of a moveable column lacks ``<column>_moved_from`` (its type, nullable) or
    ``<column>_moved_epoch`` (BIGINT, nullable), if a cache's table or the schema's columns carry them (no entity or
    INSERT names them), if the restore declaration does not name the moveable column, or if an unsound declaration is
    let through."""
    for statements in (_WIDGETS.create_statements(yugabyte=True), _WIDGETS.create_statements(yugabyte=False)):
        create = statements[0]
        assert f"{_MOVED_FROM} UUID" in create and f"{_MOVED_EPOCH} BIGINT" in create, create
        assert f"{_MOVED_FROM} UUID NOT NULL" not in create and f"{_MOVED_EPOCH} BIGINT NOT NULL" not in create
    assert json.loads(_declaration(_WIDGETS))["moveable"] == [_TENANT]
    assert {column.name for column in _WIDGETS.columns}.isdisjoint({_MOVED_FROM, _MOVED_EPOCH})
    cache = _WIDGETS.to_sqlalchemy_table(sa.MetaData())
    assert _MOVED_FROM not in cache.c and _MOVED_EPOCH not in cache.c
    with pytest.raises(ValueError, match="not ROW_STAMPED"):
        TableSchema(
            name="moving_unstamped",
            primary_key="id",
            columns=[Column("id", STRING_TYPE), Column(_TENANT, UUID_TYPE)],
            moveable=(_TENANT,),
        )
    with pytest.raises(ValueError, match="key column"):
        TableSchema(
            name="moving_by_key",
            primary_key="id",
            columns=[Column("id", STRING_TYPE)],
            row_epoch=ROW_STAMPED,
            moveable=("id",),
        )


def test_a_move_is_recorded_in_the_same_statement_that_moves_the_row() -> None:
    """Flips if a write that sets a moveable column -- the upsert's ``DO UPDATE``, the fenced ``UPDATE``, a keyed
    update's values -- does not set ``moved_from`` to the row's OLD value and ``moved_epoch`` to the row's new epoch,
    both only when the column's value changes, in that same statement (no second statement, no read)."""
    row = {"id": "w1", _TENANT: _NEW, "label": "x", "rows": 1, "date_updated": datetime.now(UTC), EPOCH_COLUMN: 7}
    upsert = schema_sql.build_insert_sql(_WIDGETS, row)
    moving = f"{_TABLE}.{_TENANT} IS DISTINCT FROM EXCLUDED.{_TENANT}"
    assert f"{_MOVED_FROM} = CASE WHEN {moving} THEN {_TABLE}.{_TENANT} ELSE {_TABLE}.{_MOVED_FROM} END" in upsert
    assert (
        f"{_MOVED_EPOCH} = CASE WHEN {moving} THEN GREATEST({_TABLE}.epoch + 1, EXCLUDED.epoch) "
        f"ELSE {_TABLE}.{_MOVED_EPOCH} END"
    ) in upsert, upsert
    assert upsert.count("INSERT") == 1 and "SELECT" not in upsert

    fenced = schema_sql.build_cas_update_sql(_WIDGETS, row)
    tenant_param = re.search(rf"{_TENANT} = (\$\d+(?:::\w+)?)", fenced)
    epoch_param = re.search(r"epoch = GREATEST\(epoch \+ 1, (\$\d+(?:::\w+)?)\)", fenced)
    assert tenant_param is not None and epoch_param is not None, fenced
    new, stamp = tenant_param.group(1), epoch_param.group(1)
    assert (
        f"{_MOVED_FROM} = CASE WHEN {_TENANT} IS DISTINCT FROM {new} THEN {_TENANT} ELSE {_MOVED_FROM} END"
    ) in fenced, fenced
    assert (
        f"{_MOVED_EPOCH} = CASE WHEN {_TENANT} IS DISTINCT FROM {new} THEN GREATEST(epoch + 1, {stamp}) "
        f"ELSE {_MOVED_EPOCH} END"
    ) in fenced, fenced
    assert fenced.startswith(f"UPDATE {_TABLE} SET ") and "SELECT" not in fenced

    values = stamped_values(_WIDGETS, {_TENANT: _NEW})
    stamp_value = values[EPOCH_COLUMN].params[0]
    assert values[_MOVED_FROM].sql == (
        f'CASE WHEN "{_TENANT}" IS DISTINCT FROM $1::uuid THEN "{_TENANT}" ELSE "{_MOVED_FROM}" END'
    )
    assert values[_MOVED_FROM].params == (_NEW,)
    assert values[_MOVED_EPOCH].sql == (
        f'CASE WHEN "{_TENANT}" IS DISTINCT FROM $1::uuid THEN GREATEST("epoch" + 1, $2::bigint) '
        f'ELSE "{_MOVED_EPOCH}" END'
    )
    assert values[_MOVED_EPOCH].params == (_NEW, stamp_value), "the move's epoch is the update's own rising stamp"


def test_a_write_that_does_not_set_the_column_leaves_the_move_alone() -> None:
    """Flips if a write that names no moveable column touches ``moved_from`` or ``moved_epoch`` (they keep the last
    move; WHY never cleared: the module's "The write"), or if a write that names one but keeps its value would record a
    move: the CASE keeps the stored values unless the value is DISTINCT."""
    row = {"id": "w1", "rows": 2, "date_updated": datetime.now(UTC), EPOCH_COLUMN: 8}
    for sql in (schema_sql.build_insert_sql(_WIDGETS, row), schema_sql.build_cas_update_sql(_WIDGETS, row)):
        assert _MOVED_FROM not in sql and _MOVED_EPOCH not in sql, sql
    assert _MOVED_FROM not in stamped_values(_WIDGETS, {"rows": 2})
    assert (
        schema_sql.moving_columns(_WIDGETS, row) == [] and schema_sql.moving_columns(_WIDGETS, row, fenced=True) == []
    )


def _store(returned: dict[str, Any] | None) -> tuple[SqlL3Backend, list[str]]:
    """a store over a stand-in pool: every statement recorded; a write that returns answers ``returned``."""
    store: Any = SqlL3Backend(MagicMock())
    sent: list[str] = []

    async def fetch(query: str, *_params: Any) -> list[dict[str, Any]]:
        sent.append(query)
        return [] if returned is None or not query.startswith(("INSERT", "UPDATE")) else [dict(returned)]

    async def fetchrow(query: str, *_params: Any) -> dict[str, Any] | None:
        sent.append(query)
        return None

    async def execute(query: str, *_params: Any) -> str:
        sent.append(query)
        return "INSERT 0 1"

    store.fetch, store.fetchrow, store.execute = fetch, fetchrow, execute
    return store, sent


async def _record_keys(bus: FakeNatsClient) -> list[str]:
    log = bus.change_log(EPOCH_CHANGES_BUCKET)
    return [entry.key for entry in await log.records_after(f"*.{_TABLE}.e", 0, 100)]


async def test_a_write_that_moves_the_row_names_both_tenants_in_its_live_record() -> None:
    """Flips if the epoch record of a write that moved a row from one tenant to another names only the new tenant (the
    old tenant's followers would keep the row: the ring's old-row note was all that told them), if the move is read in
    a second statement rather than returned by the write itself, or if the row's two messages are not both counted."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    await source.start_log()
    registry = CollectionRegistry()
    store, sent = _store({EPOCH_COLUMN: 99, _MOVED_FROM: _OLD, _MOVED_EPOCH: 99})
    registry.configure(l3_pool=store)
    registry.set_generation_source(source)
    widgets = collection_for_schema(_WIDGETS, entity_class=_Widget)(registry, DefaultCoreConfig(), None)

    await widgets.save_entity(
        widgets.create({"id": "w1", _TENANT: _NEW, "label": "x", "rows": 1, "date_updated": datetime.now(UTC)})
    )

    writes = [query for query in sent if query.startswith(("INSERT", "UPDATE"))]
    assert len(writes) == 1 and writes[0].endswith(f" RETURNING epoch, {_MOVED_FROM}, {_MOVED_EPOCH}"), writes
    keys = await _record_keys(bus)
    assert sorted(keys) == sorted(
        [epoch_change_key(_TABLE, value_tenant(_NEW)), epoch_change_key(_TABLE, value_tenant(_OLD))]
    )
    assert taken_moves(_TABLE) == (), "the move is taken by the write that made it"


async def test_a_write_that_does_not_move_the_row_names_its_own_tenant_alone() -> None:
    """Flips if a write whose statement did not move the row (its ``moved_epoch`` an earlier write's) names the tenant
    an earlier move left: only the statement that moved the row speaks for the tenant it left."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    await source.start_log()
    registry = CollectionRegistry()
    store, _sent = _store({EPOCH_COLUMN: 120, _MOVED_FROM: _OLD, _MOVED_EPOCH: 99})
    registry.configure(l3_pool=store)
    registry.set_generation_source(source)
    widgets = collection_for_schema(_WIDGETS, entity_class=_Widget)(registry, DefaultCoreConfig(), None)

    await widgets.save_entity(
        widgets.create({"id": "w1", _TENANT: _NEW, "label": "y", "rows": 2, "date_updated": datetime.now(UTC)})
    )

    assert await _record_keys(bus) == [epoch_change_key(_TABLE, value_tenant(_NEW))]
