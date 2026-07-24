"""tests for ColumnDef vector column-type validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.core.data.schema import ColumnDef


class TestVectorColumnType:
    """``vector`` is an allowed column_type with a mandatory dimension."""

    def test_vector_with_dimension_is_valid(self) -> None:
        col = ColumnDef(name="embedding", column_type="vector", vector_dim=1024)
        assert col.column_type == "vector"
        assert col.vector_dim == 1024

    def test_vector_without_dimension_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="vector_dim"):
            ColumnDef(name="embedding", column_type="vector")

    def test_vector_dim_on_non_vector_column_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="vector_dim"):
            ColumnDef(name="name", column_type="text", vector_dim=1024)

    def test_vector_dim_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ColumnDef(name="embedding", column_type="vector", vector_dim=0)

    def test_timestamptz_is_valid_column_type(self) -> None:
        """``timestamptz`` is in the closed set of declarable column types."""
        col = ColumnDef(name="date_created", column_type="timestamptz")
        assert col.column_type == "timestamptz"

    def test_unknown_column_type_is_rejected(self) -> None:
        """a type outside the closed set raises at construction."""
        with pytest.raises(ValidationError):
            ColumnDef(name="x", column_type="timestamptzz")

    def test_non_vector_columns_unchanged(self) -> None:
        col = ColumnDef(name="name", column_type="text")
        assert col.vector_dim is None
