"""An index wider than YugabyteDB builds is refused where it is declared, never at apply time.

YugabyteDB 2026.1 refuses an index of more than 32 columns, its ``INCLUDE`` columns counted ("cannot use more than
32 columns in an index", measured on 2026.1.0.0-b118), and a primary key is an index. A schema past that could be
written, rendered and shipped, then fail on the database mid-upgrade.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.core.collections.schema_backed import MAX_INDEX_COLUMNS, STRING_TYPE, Column, Index, TableSchema
from threetears.core.data.schema import IndexDef as DataIndexDef


def _names(count: int, prefix: str = "c") -> list[str]:
    return [f"{prefix}{n:02d}" for n in range(count)]


class TestSchemaBackedIndex:
    def test_the_limit_is_yugabytes(self) -> None:
        assert MAX_INDEX_COLUMNS == 32

    def test_an_index_at_the_limit_is_declared(self) -> None:
        index = Index("wide", *_names(20), include=tuple(_names(12, "i")))
        assert len(index.columns) + len(index.include) == 32

    def test_an_index_one_column_over_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at most 32"):
            Index("wide", *_names(33))

    def test_included_columns_count(self) -> None:
        with pytest.raises(ValueError, match="1 columns and 32 included are 33"):
            Index("wide", "c00", include=tuple(_names(32, "i")))

    def test_a_primary_key_over_the_limit_is_refused(self) -> None:
        names = _names(33)
        with pytest.raises(ValueError, match="primary_key lists 33 columns"):
            TableSchema(name="wide", primary_key=tuple(names), columns=[Column(n, STRING_TYPE) for n in names])

    def test_a_primary_key_at_the_limit_is_declared(self) -> None:
        names = _names(32)
        schema = TableSchema(name="wide", primary_key=tuple(names), columns=[Column(n, STRING_TYPE) for n in names])
        assert len(schema.pk_columns) == 32


class TestDataIndexDef:
    def test_an_index_at_the_limit_is_declared(self) -> None:
        assert DataIndexDef(name="wide", columns=_names(1), include=_names(31, "i")).name == "wide"

    def test_an_index_over_the_limit_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at most 32"):
            DataIndexDef(name="wide", columns=_names(10), include=_names(23, "i"))
