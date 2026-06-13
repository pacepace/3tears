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

    def test_non_vector_columns_unchanged(self) -> None:
        col = ColumnDef(name="name", column_type="text")
        assert col.vector_dim is None
