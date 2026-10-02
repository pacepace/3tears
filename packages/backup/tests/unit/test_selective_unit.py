"""Selection vocabulary and predicate building — the pure half of selective restore."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

import pytest

from threetears.backup.selective import RowSelection, SelectiveRestore


# parity-exempt: the asyncpg connection surface plan() reads through -- fetch and close only
class _SnapshotConnection:
    """a snapshot connection with a table of primary key ``pk``, recording the row selection it is sent.

    The catalog reads are told apart by what they select: ``attname`` is the primary-key read,
    ``column_name`` the column list. The selection is the one statement starting ``SELECT *``.
    """

    def __init__(self, pk: tuple[str, ...]) -> None:
        self.pk = pk
        self.selections: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        if query.startswith("SELECT * FROM"):
            self.selections.append((query, args))
            return []
        if "attname" in query:
            return [{"attname": column} for column in self.pk]
        return [{"column_name": column} for column in self.pk]

    async def close(self) -> None:
        return None


def _predicate(selection: RowSelection, pk: tuple[str, ...] = ("id",)) -> tuple[str, list[object]]:
    """the WHERE clause and parameters ``plan`` selects ``selection``'s rows with.

    Read off the statement the restore actually sends to the snapshot, so these tests pin the
    predicate where it takes effect rather than at a private builder.

    :param selection: the rows to restore
    :ptype selection: RowSelection
    :param pk: the table's primary key
    :ptype pk: tuple[str, ...]
    :return: the clause after ``WHERE``, and its bound parameters
    :rtype: tuple[str, list[object]]
    """
    connection = _SnapshotConnection(pk)

    async def _connect(_dsn: str) -> _SnapshotConnection:
        return connection

    asyncio.run(SelectiveRestore(connect=_connect, scratch_dsn="scratch", live_dsn="live").plan(selection))  # type: ignore[arg-type]
    [(query, params)] = connection.selections
    return query.split(" WHERE ", 1)[1], list(params)


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
