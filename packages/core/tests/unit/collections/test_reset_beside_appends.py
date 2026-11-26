"""A versioned table's reset marker and its appends in ONE transaction: a live table replaced atomically.

The contract this pins (reports replaces a race-keyed live table whole: its reset marker, then the table's new rows, in
one commit, so no reader ever sees it empty and no ``TRUNCATE`` runs):

- a versioned table takes appends after its reset marker in the transaction that wrote it, when the rows carry an epoch
  newer than the marker: a marker hides only rows older than itself;
- rows whose stamp was drawn before the marker's are refused, and so is a reset after the transaction's own rows;
- the commit is settled ONCE for the table, as a reset (the whole table), never a key at a time beside it; another
  table of the transaction is settled by key as before; another instance that appended, not held by the registry,
  drops its own table;
- a table reset in the transaction owes no delta file: a reset's record names none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.ddl.row_epoch import next_epoch

_TABLE = "live_results"


class _Versioned:
    """a versioned collection of a race-keyed table, recording how it is settled."""

    invalidation_columns: tuple[str, ...] = ()
    advances_write_generation = True
    writes_delta_files = False
    registry: Any = None

    def __init__(self) -> None:
        self.table_name = _TABLE
        self.resets = 0
        self.settled: list[list[dict[str, str]]] = []
        self.invalidated: list[list[Any]] = []
        self.dropped_whole = 0

    def key_prefix_of(self, constants: Mapping[str, Any]) -> dict[str, str]:
        prefix: dict[str, str] = {}
        for column in ("race", "id"):
            if constants.get(column) is None:
                break
            prefix[column] = str(constants[column])
        return prefix

    async def settle_prefixes(self, prefixes: Sequence[Mapping[str, str]]) -> None:
        self.settled.append([dict(prefix) for prefix in prefixes])

    async def settle_reset(self) -> None:
        self.resets += 1

    async def invalidate_cache_many(self, keys: Sequence[Any], **_: Any) -> None:
        self.invalidated.append(list(keys))

    def drop_cached_table(self) -> int:
        self.dropped_whole += 1
        return 0

    def changed_row(self, key: Any, row: Any) -> Any:
        return (key, row)


class _Other(_Versioned):
    """another table, written by key beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.table_name = "live_races"


class _NoTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _Conn:
    def transaction(self, **options: Any) -> Any:
        return _NoTransaction()


def _enrollments(collection: _Versioned, keys: Sequence[Any]) -> list[Any]:
    held: Any = collection
    return CallerTransaction.enrollments(held, keys, [None] * len(keys))


def _rows(race: str, count: int) -> list[tuple[str, int]]:
    return [(race, n) for n in range(count)]


@pytest.mark.parametrize("count", [3, 200])
async def test_appends_after_a_reset_marker_settle_the_table_once_as_a_reset(count: int) -> None:
    """Flips if appends beside a reset marker are refused, or settled a key at a time beside the reset (two epochs of
    one table for one commit), or the table is settled by prefixes rather than whole."""
    collection, other = _Versioned(), _Other()
    held: Any = collection
    async with CallerTransaction(_Conn()) as transaction:
        transaction.enroll_reset(held, marker_epoch=next_epoch())
        transaction.enroll_all(_enrollments(collection, _rows("r1", count) + _rows("r2", 2)), epoch=next_epoch())
        transaction.enroll_all(_enrollments(other, [("r1",)]))
    assert collection.resets == 1
    assert collection.settled == [] and collection.invalidated == []
    assert other.invalidated == [[("r1",)]] and other.resets == 0


async def test_another_instance_that_appended_drops_its_own_table_when_no_registry_holds_it() -> None:
    """Flips if an instance other than the one that reset the table keeps its cached rows of the old table, or
    advances the table a second time."""
    resetting, appending = _Versioned(), _Versioned()
    held: Any = resetting
    async with CallerTransaction(_Conn()) as transaction:
        transaction.enroll_reset(held, marker_epoch=next_epoch())
        transaction.enroll_all(_enrollments(appending, _rows("r1", 2)), epoch=next_epoch())
    assert resetting.resets == 1 and appending.resets == 0
    assert appending.dropped_whole == 1 and appending.invalidated == []


async def test_rows_stamped_before_the_reset_marker_are_refused() -> None:
    """Flips if rows whose stamp was drawn before the reset marker's are taken after it: the marker would hide them."""
    collection = _Versioned()
    held: Any = collection
    async with CallerTransaction(_Conn()) as transaction:
        early = next_epoch()
        transaction.enroll_reset(held, marker_epoch=next_epoch())
        with pytest.raises(ValueError, match="would hide them"):
            transaction.enroll_all(_enrollments(collection, _rows("r1", 1)), epoch=early)
        with pytest.raises(ValueError, match="one epoch"):
            transaction.enroll(held, ("r1", 0))


async def test_a_reset_after_the_transaction_s_own_rows_is_refused_as_hiding_them() -> None:
    """Flips if a reset marker written after the transaction's own appends of the table is taken: it would hide them."""
    collection = _Versioned()
    held: Any = collection
    async with CallerTransaction(_Conn()) as transaction:
        transaction.enroll_all(_enrollments(collection, _rows("r1", 1)), epoch=next_epoch())
        with pytest.raises(ValueError, match="would hide\\s+them"):
            transaction.enroll_reset(held, marker_epoch=next_epoch())
