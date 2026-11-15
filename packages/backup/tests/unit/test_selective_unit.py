"""Selection vocabulary and predicate building — the pure half of selective restore."""

from __future__ import annotations

import asyncio
import ast
import re
from pathlib import Path
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

import pytest

from threetears.backup.selective import RowSelection, SelectionTooLargeError, SelectiveRestore


# parity-exempt: the asyncpg connection surface plan() reads through -- fetch and close only
class _SnapshotConnection:
    """a snapshot connection with a table of primary key ``pk``, recording the row selection it is sent.

    The catalog reads are told apart by what they select: ``attname`` is the primary-key read (each column's type
    beside it), ``column_name`` the column list. The selection is each keyed page the restore reads, ``SELECT t.*``;
    ``rows`` is how many rows the selection matches, answered a page at a time as its ``LIMIT`` asks.
    """

    def __init__(self, pk: tuple[str, ...], rows: int = 0) -> None:
        self.pk = pk
        self.rows = rows
        self.answered = 0
        self.selections: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        if query.startswith("SELECT t.* FROM"):
            self.selections.append((query, args))
            limit = int(re.search(r"LIMIT (\d+)$", query).group(1))  # type: ignore[union-attr]
            page = [
                {column: f"{column}-{self.answered + index:06d}" for column in self.pk}
                for index in range(min(limit, self.rows - self.answered))
            ]
            self.answered += len(page)
            return page
        if "attname" in query:
            return [{"attname": column, "sql_type": "text"} for column in self.pk]
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
    # the keyed page writes the selection's predicate in its own parentheses, before the key's order
    return query.split(" WHERE (", 1)[1].rsplit(") ORDER BY ", 1)[0], list(params)


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


_REPO = Path(__file__).resolve().parents[4]


def test_plan_is_reached_only_from_an_operators_request() -> None:
    """``SelectiveRestore.plan`` (``@operator_command``) is an operator's request: nothing in 3tears builds a
    ``SelectiveRestore`` but its own module, so plan is reached only from a caller outside the packages (the hub's
    admin endpoint an operator calls)."""
    built: set[str] = set()
    for path in sorted(_REPO.glob("packages/*/src/**/*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", "")) == "SelectiveRestore"
            ):
                built.add(path.relative_to(_REPO).as_posix())
    assert built == set()


class TestBoundedRead:
    def test_a_selection_past_the_bound_is_refused_having_read_one_row_past_it(self) -> None:
        """Flips if the scratch read takes the whole selection before refusing it, or reads it in one statement."""
        connection = _SnapshotConnection(("id",), rows=5000)

        async def _connect(_dsn: str) -> _SnapshotConnection:
            return connection

        restore = SelectiveRestore(connect=_connect, scratch_dsn="scratch", live_dsn="live", max_rows=1200)  # type: ignore[arg-type]
        with pytest.raises(SelectionTooLargeError):
            asyncio.run(restore.plan(RowSelection(table="t", all_rows=True)))
        # each statement asks one row past the ones it keeps (the sign more follow): 1,201 kept, one more a statement
        assert len(connection.selections) == 3
        assert connection.answered == 1201 + len(connection.selections)

    def test_pages_past_the_first_start_after_the_last_key_read(self) -> None:
        """Flips if a page after the first rereads from the start (its cursor dropped) rather than after its last key."""
        connection = _SnapshotConnection(("id",), rows=700)

        async def _connect(_dsn: str) -> _SnapshotConnection:
            return connection

        restore = SelectiveRestore(connect=_connect, scratch_dsn="scratch", live_dsn="live", max_rows=600)  # type: ignore[arg-type]
        with pytest.raises(SelectionTooLargeError):
            asyncio.run(restore.plan(RowSelection(table="t", all_rows=True)))
        [first, second] = connection.selections
        assert "ORDER BY" in first[0] and first[1] == ()
        assert second[1][-1] == "id-000499", second
