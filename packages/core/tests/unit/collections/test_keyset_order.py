"""TableSchema.keyset_order: a filtered keyset read pages in the order of the index its filters lead."""

from __future__ import annotations

import pytest

from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, Index, TableSchema

_KEY = ("office_key", "candidate", "geo")


def _schema(*indexes: object) -> TableSchema:
    return TableSchema(
        name="results",
        primary_key=_KEY,
        columns=[
            Column("office_key", STRING_TYPE),
            Column("candidate", STRING_TYPE),
            Column("geo", STRING_TYPE),
            Column("state", STRING_TYPE),
            Column("votes", BIGINT_TYPE, nullable=True),
        ],
        indexes=tuple(indexes),  # type: ignore[arg-type]
    )


class TestTheOrderIsTheIndexTheFiltersLead:
    def test_no_filter_pages_on_the_primary_key(self) -> None:
        """Flips if a whole-table read stops paging on its key."""
        assert _schema(Index("by_state", "state", "geo", "office_key", "candidate")).keyset_order() == _KEY

    def test_a_filter_leading_an_index_that_holds_the_key_pages_on_the_rest_of_that_index(self) -> None:
        """Flips if a state's read is paged on the primary key while an index led by the state holds it."""
        schema = _schema(Index("by_state", "state", "geo", "office_key", "candidate"))
        assert schema.keyset_order({"state"}) == ("geo", "office_key", "candidate")

    def test_an_index_without_the_key_is_never_chosen(self) -> None:
        """Flips if paging on an index's columns could step past a row: without the key they are not unique."""
        assert _schema(Index("by_state", "state", "geo")).keyset_order({"state"}) == _KEY

    @pytest.mark.parametrize(
        "index",
        [
            Index("partial", "state", "geo", "office_key", "candidate", where="votes > 0"),
            Index("descending", "state", "geo", "office_key", "candidate", desc="geo"),
            Index("gin", "state", "geo", "office_key", "candidate", using="gin"),
        ],
        ids=["partial", "descending", "another-method"],
    )
    def test_an_index_that_cannot_answer_the_order_is_never_chosen(self, index: object) -> None:
        """Flips if a partial, descending or non-btree index is taken to answer an ascending keyset page."""
        assert _schema(index).keyset_order({"state"}) == _KEY

    def test_a_filter_leading_the_key_pages_on_the_rest_of_the_key(self) -> None:
        """Flips if a key-led read keeps paging on the column its filter already fixes."""
        assert _schema().keyset_order({"office_key"}) == ("candidate", "geo")

    def test_the_key_wins_a_tie_and_the_furthest_lead_wins(self) -> None:
        """Flips if an index is preferred to the key it ties, or a shorter lead to a longer one."""
        tied = _schema(Index("by_office", "office_key", "state", "candidate", "geo"))
        assert tied.keyset_order({"office_key"}) == ("candidate", "geo")
        further = _schema(Index("by_office_state", "office_key", "state", "candidate", "geo"))
        assert further.keyset_order({"office_key", "state"}) == ("candidate", "geo")

    def test_a_filter_pinning_the_whole_key_pages_on_the_key(self) -> None:
        """Flips if a read of one row is given nothing to page on."""
        assert _schema().keyset_order(set(_KEY)) == _KEY

    def test_a_covering_index_includes_columns_it_does_not_order_by(self) -> None:
        """Flips if an index's included columns are taken for its order, or an included column repeats one."""
        schema = _schema(Index("by_state", "state", "geo", "office_key", "candidate", include=("votes",)))
        assert schema.keyset_order({"state"}) == ("geo", "office_key", "candidate")
        with pytest.raises(ValueError, match="already its columns"):
            Index("twice", "state", "geo", include=("geo",))
