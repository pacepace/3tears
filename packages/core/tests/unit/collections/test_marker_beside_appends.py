"""A versioned table's run markers and its appends in ONE transaction: a run replaced atomically.

The contract this pins (reports replaces an LSF state's run: its run marker, then the run's new rows, in one commit, so
no reader ever sees the state with no run; "stale OK, missing is P0"):

- a versioned table takes appends after its run markers in the transaction that wrote them, when the rows carry an
  epoch newer than the newest marker of the table: a marker hides only rows older than itself, so they stay visible;
- rows whose stamp was drawn before the marker's, enrolled after it, are refused: the marker would hide them;
- a marker after the transaction's own rows of the table is refused, with an error saying it would hide them;
- a write naming no epoch (a stamped or plain table's, a single save) beside a prefix delete is still refused;
- the commit is settled ONCE for the table, by the markers' prefixes and then each prefix the rows fall in (in the
  markers' shapes), never a key at a time beside it: one epoch of the table, as the hub's broker records the commit;
  another instance that appended drops its own cached rows of that reach. Small and large commits alike (the large
  one groups its enrollments on a thread).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.ddl.row_epoch import next_epoch

_TABLE = "lsf_projections"


class _Versioned:
    """a versioned collection of a table keyed ``(state, run_id, geo)``, recording how it is settled."""

    invalidation_columns: tuple[str, ...] = ()
    advances_write_generation = True
    writes_delta_files = False

    def __init__(self) -> None:
        self.table_name = _TABLE
        self.settled: list[list[dict[str, str]]] = []
        self.invalidated: list[list[Any]] = []
        self.dropped: list[list[dict[str, str]]] = []

    def key_prefix_of(self, constants: Mapping[str, Any]) -> dict[str, str]:
        prefix: dict[str, str] = {}
        for column in ("state", "run_id", "geo"):
            if constants.get(column) is None:
                break
            prefix[column] = str(constants[column])
        return prefix

    async def settle_prefixes(self, prefixes: Sequence[Mapping[str, str]]) -> None:
        self.settled.append([dict(prefix) for prefix in prefixes])

    async def invalidate_cache_many(self, keys: Sequence[Any], **_: Any) -> None:
        self.invalidated.append(list(keys))

    async def drop_cached_prefixes(self, prefixes: Sequence[Mapping[str, str]]) -> int:
        self.dropped.append([dict(prefix) for prefix in prefixes])
        return 0

    def changed_row(self, key: Any, row: Any) -> Any:
        return (key, row)


class _Other(_Versioned):
    """another table, written by key beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.table_name = "lsf_runs"


class _NoTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _Conn:
    def transaction(self, **options: Any) -> Any:
        return _NoTransaction()


_MARKED = [{"state": "PA", "run_id": "7"}]


def _any(collection: _Versioned) -> Any:
    """the stand-in, as the collection a write enrolls."""
    return collection


def _enrollments(collection: _Versioned, keys: Sequence[Any], rows: Sequence[Any]) -> list[Any]:
    return CallerTransaction.enrollments(_any(collection), keys, rows)


def _rows(run: int, count: int) -> list[tuple[str, int, str]]:
    return [("PA", run, f"g{n:04d}") for n in range(count)]


@pytest.mark.parametrize("count", [3, 200])
async def test_appends_after_a_run_marker_settle_the_table_once_by_the_markers_and_the_rows_prefixes(
    count: int,
) -> None:
    """Flips if appends beside a run marker are refused, or settled a key at a time beside the prefixes (two epochs of
    one table for one commit), or the record leaves out the prefixes the rows fall in."""
    collection, other = _Versioned(), _Other()
    async with CallerTransaction(_Conn()) as transaction:
        marker = next_epoch()
        transaction.enroll_prefixes(collection, _MARKED, marker_epoch=marker)
        keys = _rows(8, count) + _rows(9, 2)
        transaction.enroll_all(_enrollments(collection, keys, [None] * len(keys)), epoch=next_epoch())
        transaction.enroll_all(_enrollments(other, [("r1",)], [None]))
    assert collection.settled == [
        [{"state": "PA", "run_id": "7"}, {"state": "PA", "run_id": "8"}, {"state": "PA", "run_id": "9"}]
    ]
    assert collection.invalidated == []
    # another table of the transaction is settled by key, as before
    assert other.invalidated == [[("r1",)]]


async def test_another_instance_that_appended_drops_its_own_cached_rows_of_the_reach() -> None:
    """Flips if an instance other than the one that wrote the markers keeps its cached rows of the replaced run."""
    marking, appending = _Versioned(), _Versioned()
    async with CallerTransaction(_Conn()) as transaction:
        transaction.enroll_prefixes(marking, _MARKED, marker_epoch=next_epoch())
        transaction.enroll_all(_enrollments(appending, _rows(8, 2), [None, None]), epoch=next_epoch())
    reach = [{"state": "PA", "run_id": "7"}, {"state": "PA", "run_id": "8"}]
    assert marking.settled == [reach]
    assert appending.dropped == [reach] and appending.invalidated == []


async def test_rows_stamped_before_the_marker_are_refused() -> None:
    """Flips if rows whose stamp was drawn before the marker's are taken after it: the marker would hide them."""
    collection = _Versioned()
    async with CallerTransaction(_Conn()) as transaction:
        early = next_epoch()
        transaction.enroll_prefixes(collection, _MARKED, marker_epoch=next_epoch())
        with pytest.raises(ValueError, match="would hide them"):
            transaction.enroll_all(_enrollments(collection, _rows(8, 1), [None]), epoch=early)


async def test_a_marker_after_the_transaction_s_own_rows_is_refused_as_hiding_them() -> None:
    """Flips if a run marker written after the transaction's own appends of the table is taken (it would hide them),
    or refused with the plain one-kind-of-write message that does not say why."""
    collection = _Versioned()
    async with CallerTransaction(_Conn()) as transaction:
        transaction.enroll_all(_enrollments(collection, _rows(8, 1), [None]), epoch=next_epoch())
        with pytest.raises(ValueError, match="would hide\\s+them"):
            transaction.enroll_prefixes(collection, _MARKED, marker_epoch=next_epoch())


async def test_a_write_naming_no_epoch_beside_a_prefix_delete_is_still_refused() -> None:
    """Flips if a stamped or plain table's write (no epoch), or a single keyed save, is taken beside a prefix delete of
    its table: only a versioned table's appends after its markers are."""
    collection = _Versioned()
    async with CallerTransaction(_Conn()) as transaction:
        transaction.enroll_prefixes(collection, _MARKED, marker_epoch=next_epoch())
        with pytest.raises(ValueError, match="one epoch"):
            transaction.enroll_all(_enrollments(collection, _rows(8, 1), [None]))
        with pytest.raises(ValueError, match="one epoch"):
            transaction.enroll(_any(collection), ("PA", 8, "g0"))
    plain = _Versioned()
    async with CallerTransaction(_Conn()) as transaction:
        # a delete with no marker epoch: a plain table's prefixes are deleted, never marked
        transaction.enroll_prefixes(plain, _MARKED)
        with pytest.raises(ValueError, match="one epoch"):
            transaction.enroll_all(_enrollments(plain, _rows(8, 1), [None]), epoch=next_epoch())
