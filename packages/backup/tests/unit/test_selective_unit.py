"""Selection vocabulary and predicate building — the pure half of selective restore."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid7

import pytest

from threetears.backup.selective import RowSelection, SelectiveRestore


def _predicate(selection: RowSelection, pk: tuple[str, ...] = ("id",)):
    restore = SelectiveRestore(connect=None, scratch_dsn="", live_dsn="")  # type: ignore[arg-type]
    return restore._predicate(selection, pk)  # noqa: SLF001 -- the pure predicate builder is worth pinning directly


class TestRowSelection:
    def test_exactly_one_vocabulary_is_required(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            RowSelection(table="t")
        with pytest.raises(ValueError, match="exactly one"):
            RowSelection(table="t", ids=(uuid7(),), all_rows=True)

    def test_an_empty_id_list_is_refused_not_a_noop(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            RowSelection(table="t", ids=())


class TestPredicates:
    def test_explicit_ids_bind_as_an_array(self) -> None:
        ids = (uuid7(), uuid7())
        where, params = _predicate(RowSelection(table="t", ids=ids))
        assert where == '"id" = ANY($1)'
        assert params == [list(ids)]

    def test_id_range_is_inclusive_both_ends(self) -> None:
        low, high = uuid7(), uuid7()
        where, params = _predicate(RowSelection(table="t", id_range=(low, high)))
        assert where == '"id" >= $1 AND "id" <= $2'
        assert params == [low, high]

    def test_date_range_addresses_the_named_column(self) -> None:
        low = datetime(2026, 9, 1, tzinfo=UTC)
        high = datetime(2026, 9, 2, tzinfo=UTC)
        where, params = _predicate(RowSelection(table="t", date_range=("date_created", low, high)))
        assert where == '"date_created" >= $1 AND "date_created" <= $2'
        assert params == [low, high]

    def test_composite_pk_demands_an_explicit_id_column(self) -> None:
        with pytest.raises(ValueError, match="composite primary key"):
            _predicate(RowSelection(table="t", ids=(uuid7(),)), pk=("customer_id", "id"))

    def test_composite_pk_with_named_id_column_works(self) -> None:
        where, _ = _predicate(RowSelection(table="t", ids=(uuid7(),), id_column="id"), pk=("customer_id", "id"))
        assert where == '"id" = ANY($1)'

    def test_unsafe_identifiers_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            _predicate(
                RowSelection(table="t", date_range=('x"; DROP TABLE t; --', datetime.now(UTC), datetime.now(UTC)))
            )


class TestRawWhere:
    def test_a_raw_predicate_passes_through_with_its_params(self) -> None:
        where, params = _predicate(
            RowSelection(table="t", where="status = $1 AND customer_id = $2", where_params=("borked", 7))
        )
        assert where == "status = $1 AND customer_id = $2"
        assert params == ["borked", 7]

    def test_where_is_one_vocabulary_among_the_five(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            RowSelection(table="t", where="TRUE", all_rows=True)

    def test_a_blank_where_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be blank"):
            RowSelection(table="t", where="   ")
