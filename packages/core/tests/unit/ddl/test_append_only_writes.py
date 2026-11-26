"""append-only writes of a versioned table (``design-append-only-l3.md``, chunk 2): the statements and plans a store
builds for one, before anything is sent.

A versioned table (``row_versions``) is never upserted and never deleted from: a write's statement names no conflict
clause, a delete is a "gone" marker (the key, the scope, the epoch and the flag) appended the same way, a whole run
deleted is ONE run marker. The table declares its gone flag, and its required columns a marker leaves out are required
of a live version by one ``CHECK``. A stamped table (``ROW_STAMPED``) keeps its upsert, one row a key. The live tests
(``tests/integration/test_append_only_writes_live.py``) prove the same on Postgres and YugabyteDB.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.schema_sql import build_bulk_insert_sql, build_insert_sql
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
)
from threetears.core.ddl.row_epoch import (
    EPOCH_COLUMN,
    GONE_COLUMN,
    ROW_STAMPED,
    AppendOnlyWriteError,
    UnstampedWriteError,
    row_versions,
)
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE, run_marker_insert_sql
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_CANDIDATES = TableSchema(
    name="ve_results",
    primary_key=("office_key", "candidate"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("join_state_code", STRING_TYPE),
        Column("votes", INT_TYPE),
        Column("share", INT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

#: an LSF table: its key leads with the scope, then the run
_PROJECTIONS = TableSchema(
    name="ve_projections",
    primary_key=("join_state_code", "run_id", "geo_id"),
    columns=[
        Column("join_state_code", STRING_TYPE),
        Column("run_id", INT_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("value", INT_TYPE),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

_RUNS = TableSchema(
    name="runs",
    primary_key="run_id",
    columns=[Column("run_id", STRING_TYPE), Column("state", STRING_TYPE, nullable=True)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_ROW = {
    "office_key": "sn_NC1",
    "candidate": "a",
    "join_state_code": "NC",
    "votes": 1,
    "share": 2,
    "date_created": None,
    "date_updated": None,
    EPOCH_COLUMN: 5 << 31,
}


def _store(*schemas: TableSchema, pool: Any = None) -> SqlL3Backend:
    store = SqlL3Backend(pool or MagicMock())
    for schema in schemas:
        store.register_schema(schema.name, schema)
    return store


# -- the DDL: the gone flag, and required columns required of a live version ------------------------------------------


def test_a_versioned_table_declares_its_gone_flag_false_unless_written() -> None:
    gone = _CANDIDATES.column(GONE_COLUMN)
    assert (gone.column_type, gone.nullable, gone.server_default, gone.immutable) == (BOOL_TYPE, False, "false", True)
    assert _CANDIDATES.versioned and not _RUNS.versioned
    assert _RUNS.get_column(GONE_COLUMN) is None


def test_a_gone_marker_carries_the_key_the_scope_the_epoch_and_the_flag() -> None:
    assert _CANDIDATES.gone_marker_columns == ("office_key", "candidate", "join_state_code", EPOCH_COLUMN, GONE_COLUMN)
    # the scope in the key is carried once
    assert _PROJECTIONS.gone_marker_columns == ("join_state_code", "run_id", "geo_id", EPOCH_COLUMN, GONE_COLUMN)
    assert _RUNS.gone_marker_columns == ()


@pytest.mark.parametrize("yugabyte", [False, True])
def test_a_versioned_tables_required_columns_are_required_of_a_live_version_by_one_check(yugabyte: bool) -> None:
    assert _CANDIDATES.live_only_columns == ("votes", "date_created", "date_updated")
    create, _index, _comment = _CANDIDATES.create_statements(yugabyte=yugabyte)
    assert "gone BOOLEAN DEFAULT false NOT NULL" in create
    assert (
        'CONSTRAINT ve_results_live CHECK ("gone" OR ("votes" IS NOT NULL AND "date_created" IS NOT NULL AND ' in create
    )
    # a marker leaves them out, so the stored table holds them nullable; the key, scope and epoch stay required
    for column in (
        "votes INTEGER,",
        "date_created TIMESTAMP WITH TIME ZONE,",
        "date_updated TIMESTAMP WITH TIME ZONE,",
    ):
        assert column in create
    for column in ("join_state_code TEXT NOT NULL", "epoch BIGINT NOT NULL", "office_key TEXT NOT NULL"):
        assert column in create


def test_a_versioned_tables_cache_keeps_its_required_columns_required() -> None:
    table = _CANDIDATES.to_sqlalchemy_table(MetaData())
    assert not table.columns["votes"].nullable
    assert not [constraint for constraint in table.constraints if constraint.name == "ve_results_live"]


def test_the_run_marker_table_is_keyed_by_table_scope_prefix_and_epoch() -> None:
    assert RUN_MARKERS_SCHEMA.name == RUN_MARKERS_TABLE
    assert RUN_MARKERS_SCHEMA.pk_columns == ("table_name", "scope", "prefix", EPOCH_COLUMN)
    [create] = RUN_MARKERS_SCHEMA.create_statements(yugabyte=True)
    assert "PRIMARY KEY (table_name, scope, prefix, epoch)" in create


# -- a write appends, never an upsert ---------------------------------------------------------------------------------


def test_a_versioned_tables_insert_names_no_conflict_clause() -> None:
    assert "ON CONFLICT" not in build_insert_sql(_CANDIDATES, dict(_ROW))
    assert "ON CONFLICT" not in build_bulk_insert_sql(_CANDIDATES, rows=3)


def test_a_stamped_tables_write_stays_an_upsert_of_one_row_a_key_setting_its_epoch() -> None:
    sql = build_insert_sql(_RUNS, {"run_id": "r1", "state": "NC", EPOCH_COLUMN: 1})
    assert "ON CONFLICT (run_id) DO UPDATE SET" in sql
    # never lower than the row's own: each writing process keeps its own clock (the HLC, test_hlc_epoch)
    assert f"{EPOCH_COLUMN} = GREATEST(runs.{EPOCH_COLUMN} + 1, EXCLUDED.{EPOCH_COLUMN})" in sql


def test_an_arrow_plan_of_a_versioned_table_appends_its_versions_and_its_gone_markers() -> None:
    store = _store(_CANDIDATES)
    conn = MagicMock(write_arrow=True)
    markers = store.arrow_write_plan(
        "ve_results",
        list(_CANDIDATES.gone_marker_columns),
        [pa.string(), pa.string(), pa.string(), pa.int64(), pa.bool_()],
        kind="gone",
        conn=conn,
    )
    assert markers is not None
    assert markers.kind == "append"
    assert [name for name, _, _ in markers.columns] == list(_CANDIDATES.gone_marker_columns)
    with pytest.raises(KeyError, match="join_state_code"):
        store.arrow_write_plan(
            "ve_results",
            ["office_key", "candidate", EPOCH_COLUMN, GONE_COLUMN],
            [pa.string(), pa.string(), pa.int64(), pa.bool_()],
            kind="gone",
            conn=conn,
        )
    with pytest.raises(AppendOnlyWriteError):
        store.arrow_write_plan(
            "ve_results", ["office_key", "candidate"], [pa.string(), pa.string()], kind="delete", conn=conn
        )


async def test_an_arrow_append_goes_through_the_door_as_an_append() -> None:
    store = _store(_CANDIDATES)
    conn = MagicMock(write_arrow=True)
    conn.write_arrow_rows = AsyncMock(return_value=1)
    rows = pa.table(
        {
            "office_key": ["sn_NC1"],
            "candidate": ["a"],
            "join_state_code": ["NC"],
            EPOCH_COLUMN: pa.array([1], pa.int64()),
            GONE_COLUMN: [True],
        }
    )
    plan = store.arrow_write_plan("ve_results", rows.column_names, rows.schema.types, kind="gone", conn=conn)
    assert plan is not None
    await store.write_arrow("ve_results", rows, plan, conn=conn)
    kwargs = conn.write_arrow_rows.await_args.kwargs
    assert kwargs["append"] is True and kwargs["delete"] is False and kwargs["conflict_columns"] == ()


# -- a delete appends a gone marker, never a DELETE -------------------------------------------------------------------


async def test_a_versioned_table_is_never_deleted_from() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()
    store = _store(_CANDIDATES, pool=pool)
    with pytest.raises(AppendOnlyWriteError):
        await store.delete_many("ve_results", [("sn_NC1", "a")], max_rows=10)
    with pytest.raises(AppendOnlyWriteError):
        await store.delete_many_returning("ve_results", [("sn_NC1", "a")], max_rows=10)
    with pytest.raises(AppendOnlyWriteError):
        await store.delete("ve_results", {"office_key": "sn_NC1", "candidate": "a"})
    pool.execute.assert_not_called()


async def test_a_gone_marker_is_an_insert_of_the_key_the_scope_and_a_newer_epoch() -> None:
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="INSERT 0 2")
    store = _store(_CANDIDATES)
    appended = await store.mark_gone(
        "ve_results", [("sn_NC1", "a"), ("sn_NC1", "b")], epoch=9 << 31, scope="NC", max_rows=10, conn=conn
    )
    assert appended == 2
    [call] = conn.execute.await_args_list
    sql, *params = call.args
    assert sql.startswith("INSERT INTO ve_results (office_key, candidate, join_state_code, epoch, gone) VALUES ($1")
    assert "ON CONFLICT" not in sql and "DELETE" not in sql
    assert params == ["sn_NC1", "a", "NC", 9 << 31, True, "sn_NC1", "b", "NC", 9 << 31, True]
    with pytest.raises(ValueError, match="name the deletes' scope"):
        await store.mark_gone("ve_results", [("sn_NC1", "a")], epoch=1, scope=None, max_rows=10, conn=conn)


# -- a whole run deleted is ONE run marker ----------------------------------------------------------------------------


def test_a_run_marker_casts_its_prefix_to_text_by_the_database() -> None:
    sql = run_marker_insert_sql(["text", "integer"], 0)
    assert sql == (
        "INSERT INTO threetears_run_markers (table_name, scope, prefix, epoch) VALUES "
        "($1, ($3::text)::text, array_to_string(ARRAY[($3::text)::text, ($4::integer)::text], chr(31)), $2)"
    )
    assert "to_char(" in run_marker_insert_sql(["timestamp with time zone"], 0)


async def test_a_whole_run_of_a_versioned_table_deleted_is_one_run_marker_a_prefix() -> None:
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchrow = AsyncMock(return_value={"whole": None, "reach": None, "guard": "g", "noted": None})
    store = _store(_PROJECTIONS)
    marked = await store.delete_prefixes(
        "ve_projections",
        [
            {"join_state_code": "PA", "run_id": 7},
            {"join_state_code": "PA", "run_id": 8},
            {"join_state_code": "PA", "run_id": 7},
        ],
        conn=conn,
        epoch=3 << 31,
    )
    assert marked == 2
    statements = [call.args for call in conn.execute.await_args_list]
    inserts = [args for args in statements if args[0].startswith("INSERT INTO threetears_run_markers")]
    assert [args[1:] for args in inserts] == [
        ("ve_projections", 3 << 31, "PA", 7),
        ("ve_projections", 3 << 31, "PA", 8),
    ]
    # nothing deleted: the one statement on the table is the no-row one that leaves its whole note
    assert [args[0] for args in statements if "ve_projections" in args[0] and args[0].startswith("DELETE")] == [
        "DELETE FROM ve_projections WHERE FALSE"
    ]


async def test_a_run_delete_of_a_versioned_table_names_its_epoch_and_its_scope() -> None:
    conn = MagicMock()
    store = _store(_PROJECTIONS, _RUNS)
    with pytest.raises(UnstampedWriteError, match="name it"):
        await store.delete_prefixes("ve_projections", [{"join_state_code": "PA"}], conn=conn)
    with pytest.raises(ValueError, match="does not name it"):
        # a run of a table whose key does not lead with its scope: a marker could not be found by it
        await _store(_CANDIDATES).delete_prefixes("ve_results", [{"office_key": "sn_NC1"}], conn=conn, epoch=1)
    with pytest.raises(ValueError, match="is not one"):
        await store.delete_prefixes("runs", [{"run_id": "r1"}], conn=conn, epoch=1)
