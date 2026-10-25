"""the generated update rewrites only the columns its row names (v0.78.0).

A partial save used to write ``NULL`` to every nullable column without a server default that its
row left out: the ``DO UPDATE SET`` and the fenced ``UPDATE`` both named every mutable column. The
real-Postgres proof is ``tests/integration/test_partial_update_keeps_columns_live.py``; these pin
the statements' shape.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from threetears.core.backends import schema_sql
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
)

_NOW = datetime(2026, 10, 25, tzinfo=UTC)


def _schema(*, cas_null_safe: bool = False) -> TableSchema:
    return TableSchema(
        name="sources",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("geo", STRING_TYPE, nullable=True),
            Column("rank", INT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        cas_null_safe=cas_null_safe,
    )


class TestTheUpsert:
    def test_its_update_sets_only_the_columns_the_row_names(self) -> None:
        row = {"id": uuid.uuid4(), "name": "n", "date_created": _NOW, "date_updated": _NOW}
        sql = schema_sql.build_insert_sql(_schema(), row)
        assert sql.endswith("DO UPDATE SET name = EXCLUDED.name, date_updated = EXCLUDED.date_updated")

    def test_its_insert_still_names_every_column_so_a_new_row_takes_null(self) -> None:
        row = {"id": uuid.uuid4(), "name": "n", "date_created": _NOW}
        sql = schema_sql.build_insert_sql(_schema(), row)
        assert sql.startswith("INSERT INTO sources (id, name, geo, rank, date_created, date_updated) VALUES")
        assert schema_sql.build_insert_params(_schema(), row)[2:4] == [None, None]

    def test_a_column_named_as_none_is_set(self) -> None:
        row = {"id": uuid.uuid4(), "name": "n", "geo": None, "date_created": _NOW}
        assert "DO UPDATE SET name = EXCLUDED.name, geo = EXCLUDED.geo" in schema_sql.build_insert_sql(_schema(), row)

    def test_a_row_naming_no_mutable_column_updates_the_row_to_itself(self) -> None:
        sql = schema_sql.build_insert_sql(_schema(), {"id": uuid.uuid4(), "date_created": _NOW})
        assert sql.endswith("DO UPDATE SET name = sources.name")

    def test_no_row_sets_every_mutable_column(self) -> None:
        sql = schema_sql.build_insert_sql(_schema())
        assert sql.endswith(
            "DO UPDATE SET name = EXCLUDED.name, geo = EXCLUDED.geo, rank = EXCLUDED.rank, "
            "date_updated = EXCLUDED.date_updated"
        )

    def test_a_bulk_upsert_still_sets_every_mutable_column(self) -> None:
        columns = schema_sql.insert_columns_for_data(_schema(), {})
        sql = schema_sql.build_bulk_insert_sql(_schema(), rows=2, columns=columns)
        assert "geo = EXCLUDED.geo, rank = EXCLUDED.rank" in sql


class TestTheFencedUpdate:
    def test_it_sets_only_the_columns_the_row_names_and_its_params_align(self) -> None:
        source_id = uuid.uuid4()
        row = {"id": source_id, "geo": None, "date_updated": _NOW}
        sql = schema_sql.build_cas_update_sql(_schema(), row)
        assert sql == "UPDATE sources SET geo = $2, date_updated = $3 WHERE id = $1 AND date_updated = $4"
        assert schema_sql.build_cas_params(_schema(), row, _NOW) == [source_id, None, _NOW, _NOW]

    def test_a_row_naming_no_mutable_column_still_matches_the_fence(self) -> None:
        sql = schema_sql.build_cas_update_sql(_schema(), {"id": uuid.uuid4()})
        assert sql == "UPDATE sources SET date_updated = date_updated WHERE id = $1 AND date_updated = $2"


class TestTheNullSafeUpsert:
    def test_its_update_sets_only_the_columns_the_row_names(self) -> None:
        row = {"id": uuid.uuid4(), "rank": 3, "date_updated": _NOW}
        sql = schema_sql.build_cas_upsert_sql(_schema(cas_null_safe=True), row)
        assert "DO UPDATE SET rank = EXCLUDED.rank, date_updated = EXCLUDED.date_updated WHERE" in sql

    def test_a_row_leaving_out_the_fence_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="advance the fence"):
            schema_sql.build_cas_upsert_sql(_schema(cas_null_safe=True), {"id": uuid.uuid4(), "rank": 3})
