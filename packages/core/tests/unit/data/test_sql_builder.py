"""tests for CREATE TABLE SQL generation from TableDef."""

from __future__ import annotations

from threetears.core.data.schema import ColumnDef, TableDef
from threetears.core.data.sql_builder import build_create_table_sql


class TestVectorDdl:
    """vector columns render as ``VECTOR(dim)`` in PostgreSQL DDL."""

    def test_vector_column_renders_dimension(self) -> None:
        table = TableDef(
            name="embeddings",
            columns=[
                ColumnDef(name="id", column_type="text", primary_key=True),
                ColumnDef(name="embedding", column_type="vector", vector_dim=1024, nullable=False),
            ],
        )

        sql = build_create_table_sql(table)

        assert "embedding VECTOR(1024) NOT NULL" in sql

    def test_non_vector_columns_unchanged(self) -> None:
        table = TableDef(
            name="widgets",
            columns=[
                ColumnDef(name="id", column_type="uuid", primary_key=True),
                ColumnDef(name="name", column_type="text"),
            ],
        )

        sql = build_create_table_sql(table)

        assert "id UUID" in sql
        assert "name TEXT" in sql
