"""a collated index column (``IndexDef.collate``) as the DDL creates it, and what a declaration may not say."""

from __future__ import annotations

from typing import Any, cast

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from threetears.core.collections.schema_backed import Index
from threetears.core.coordination.source_token import source_tokens_schema


def test_the_fence_index_is_created_c_ordered_and_range_sharded() -> None:
    """Flips if the fence's prefix index loses its C collation or its ASC (YugabyteDB would hash it), or a tuple of
    purposes is not kept as the reads it names."""
    table = source_tokens_schema().to_sqlalchemy_table(sa.MetaData())
    created = [str(CreateIndex(index).compile(dialect=cast(Any, postgresql).dialect())) for index in table.indexes]
    assert created == ['CREATE INDEX source_tokens_by_signal_c ON source_tokens (signal COLLATE "C" ASC)']
    both = Index("t_by_a", "a", purpose=("src/x.py::one", "src/x.py::two"))
    assert both.purpose == ("src/x.py::one", "src/x.py::two")


@pytest.mark.parametrize(
    "collate",
    [{"b": "C"}, {"a": "C; DROP"}],
)
def test_a_collation_names_a_column_of_the_index_and_a_plain_name(collate: dict[str, str]) -> None:
    """Flips if a collation of a column the index does not cover, or one that is not a plain name, is declared."""
    with pytest.raises(ValueError, match="collate"):
        Index("t_by_a", "a", collate=collate)
