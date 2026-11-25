"""the row epoch is a per-process hybrid logical clock (``design-append-only-l3.md``, decision 3;
``research-shared-clock.md``).

One BIGINT ``wall_ms << 15 | logical``, monotone within the process whatever its wall clock does: when the wall clock
goes back the logical part advances, and when the logical part overflows the wall part moves on by one ms. Across
processes, newest-wins rides on ``ScopeEpochs``' ``last_epoch``: the next writer's ``begin`` moves its clock past the
previous writer's highest stamp. A stamped (in-place) table's update never lowers a row's epoch. A commit older than W
is refused, and every commit names its lowest stamp (``s_min``), which its epoch records carry.
"""

from __future__ import annotations

import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import pytest

from threetears.core.backends.schema_sql import build_bulk_insert_sql, build_cas_update_sql, build_insert_sql
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.epoch_change import ChangedRow, epoch_change_body, read_epoch_change_body
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
)
from threetears.core.collections.scope_epochs import WHOLE, ScopeEpochs
from threetears.core.ddl.row_epoch import (
    EPOCH_COLUMN,
    MAX_TRANSACTION_AGE_MS,
    ROW_STAMPED,
    TransactionTooOldError,
    epoch_wall_ms,
    next_epoch,
    row_versions,
)
from threetears.core.testing.l1_tables import TEST_L1_TABLE

#: the logical part's width, by the documented encoding (``wall_ms << 15 | logical``)
_LOGICAL = 15


class _Wall:
    """a wall clock a test sets, read through ``time.time_ns`` as the process clock reads it."""

    def __init__(self, ms: int) -> None:
        self.ms = ms

    def time_ns(self) -> int:
        return self.ms * 1_000_000


@pytest.fixture
def wall(monkeypatch: pytest.MonkeyPatch) -> _Wall:
    """the wall clock, a second past the process clock's last stamp, so the process clock (which never goes back) reads
    it first. Kept close: the process clock keeps whatever lead a test gives it, for every later test in the worker."""
    clock = _Wall(epoch_wall_ms(next_epoch()) + 1_000)
    monkeypatch.setattr(time, "time_ns", clock.time_ns)
    return clock


# -- the clock ---------------------------------------------------------------------------------------------------------


def test_an_epoch_is_the_wall_millisecond_then_a_logical_count(wall: _Wall) -> None:
    epoch = next_epoch()
    assert epoch_wall_ms(epoch) == wall.ms == epoch >> _LOGICAL
    assert epoch & ((1 << _LOGICAL) - 1) == 0, "a fresh millisecond starts its logical count at 0"
    assert next_epoch() == epoch + 1, "a second stamp in the same millisecond takes the next logical count"
    assert epoch < 1 << 63


def test_the_clock_never_goes_back_when_the_wall_clock_does(wall: _Wall) -> None:
    before = next_epoch()
    wall.ms -= 10_000  # an NTP step back of ten seconds
    after = [next_epoch() for _ in range(5)]
    assert after == [before + 1 + n for n in range(5)], "the logical part advances from the last stamp"
    assert all(epoch_wall_ms(epoch) == epoch_wall_ms(before) for epoch in after)
    wall.ms += 20_000  # the wall clock passes the last stamp again: it is taken up at once
    assert next_epoch() == wall.ms << _LOGICAL


def test_a_logical_overflow_moves_the_wall_part_on_by_one_millisecond(wall: _Wall) -> None:
    first = next_epoch()
    drawn = [next_epoch() for _ in range(1 << _LOGICAL)]
    assert drawn == sorted(set(drawn)), "every stamp above the one before"
    assert epoch_wall_ms(drawn[-2]) == epoch_wall_ms(first)
    assert epoch_wall_ms(drawn[-1]) == epoch_wall_ms(first) + 1
    assert drawn[-1] & ((1 << _LOGICAL) - 1) == 0
    assert wall.ms == epoch_wall_ms(first), "the wall clock never moved: the overflow carried"


# -- a stamped (in-place) table never lowers a row's epoch ---------------------------------------------------------------

_RUNS = TableSchema(
    name="runs",
    primary_key="run_id",
    columns=[
        Column("run_id", STRING_TYPE),
        Column("state", STRING_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_RISING = "epoch = GREATEST(runs.epoch + 1, EXCLUDED.epoch)"


def test_a_stamped_tables_upsert_sets_the_epoch_to_the_greater_of_the_rows_next_and_the_new_stamp() -> None:
    row = {"run_id": "r1", "state": "done", EPOCH_COLUMN: 5}
    single = build_insert_sql(_RUNS, row)
    assert _RISING in single
    assert "epoch = EXCLUDED.epoch" not in single
    assert _RISING in build_bulk_insert_sql(_RUNS, rows=2)
    # a fenced update of the row in place, too
    assert "epoch = GREATEST(epoch + 1, $" in build_cas_update_sql(replace(_RUNS, cas_column="date_updated"), row)


def test_a_versioned_table_appends_and_never_rises_in_place() -> None:
    versioned = TableSchema(
        name="results",
        primary_key=("office_key", "candidate"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("candidate", STRING_TYPE),
            Column("join_state_code", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        row_epoch=row_versions("join_state_code"),
        l1_eviction=TEST_L1_TABLE,
    )
    assert "GREATEST" not in build_insert_sql(versioned, {"office_key": "o", "candidate": "c", EPOCH_COLUMN: 5})


# -- a transaction older than W is refused; its lowest stamp is its s_min ---------------------------------------------


class _Conn:
    """a connection whose transaction records how it ended and every statement sent."""

    def __init__(
        self, *, writing: int = 7, settled_last: int | None = None, pending: list[dict[str, Any]] | None = None
    ) -> None:
        self.writing = writing
        self.settled_last = settled_last
        self.pending = pending or []
        self.ended: list[type[BaseException] | None] = []
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    @asynccontextmanager
    async def _transaction(self) -> Any:
        try:
            yield self
        except BaseException as error:
            self.ended.append(type(error))
            raise
        self.ended.append(None)

    def transaction(self, **_: Any) -> Any:
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append((sql, params))
        return "UPDATE 1"

    async def fetchrow(self, sql: str, *params: Any) -> Any:
        self.statements.append((sql, params))
        return {"writing": self.writing, "last_epoch": self.settled_last}

    async def fetch(self, sql: str, *params: Any) -> Any:
        self.statements.append((sql, params))
        if '"pending" IS NOT NULL' in sql:
            return list(self.pending)
        return [{"writing": self.writing}]

    def last_epochs(self) -> list[tuple[Any, ...]]:
        return [params for sql, params in self.statements if "SET last_epoch" in sql]


class _EpochsTable:
    """the epochs table's collection, as ``ScopeEpochs`` and a transaction's settle reach it."""

    table_name = "scope_epochs"
    invalidation_columns = ()
    writes_delta_files = False
    advances_write_generation = False

    async def invalidate_cache_many(self, keys: Any, **_: Any) -> None:
        return None


def _epochs() -> ScopeEpochs:
    return ScopeEpochs(_EpochsTable())  # type: ignore[arg-type]


async def test_a_commit_older_than_w_is_refused_and_rolled_back(wall: _Wall) -> None:
    conn = _Conn()
    with pytest.raises(TransactionTooOldError, match="refused"):
        async with CallerTransaction(conn) as transaction:
            transaction.note_write_version(7)
            transaction.row_epoch("results")
            wall.ms += MAX_TRANSACTION_AGE_MS + 1
    assert conn.ended == [TransactionTooOldError]


async def test_a_commit_within_w_lands(wall: _Wall) -> None:
    conn = _Conn()
    async with CallerTransaction(conn) as transaction:
        transaction.note_write_version(7)
        transaction.row_epoch("results")
        wall.ms += MAX_TRANSACTION_AGE_MS
    assert conn.ended == [None]


# -- newest-wins across two processes, through last_epoch -------------------------------------------------------------

#: process A: its wall clock the milliseconds it is given (two seconds ahead of process B's), its highest stamp printed
_PROCESS_A = """
import sys, time
ahead = int(sys.argv[1])
time.time_ns = lambda: ahead * 1_000_000
from threetears.core.ddl.row_epoch import next_epoch
print(max(next_epoch() for _ in range(3)))
"""


@pytest.mark.uncacheable("process A is a second Python process: the clock under test is per process")
async def test_two_processes_writing_one_scope_stay_newest_wins_through_last_epoch() -> None:
    # process A, its clock ahead, writes scope state:NC and commits: its touch records its highest stamp on the scope
    a_wall = epoch_wall_ms(next_epoch()) + 2_000
    ahead = int(
        subprocess.run(
            [sys.executable, "-c", _PROCESS_A, str(a_wall)], capture_output=True, text=True, check=True
        ).stdout
    )
    # this process is B, its clock two seconds behind A's: a plain stamp sorts below A's
    assert next_epoch() < ahead

    # B writes the scope next. Its begin reads A's last_epoch (still pending: A's write was superseded before it
    # committed) and moves B's clock past it
    conn = _Conn(writing=8, settled_last=None, pending=[{"scope": "state:NC", "last_epoch": ahead}])
    async with CallerTransaction(conn):
        assert await _epochs().begin(conn=conn) == 8
    stamped = next_epoch()
    assert stamped > ahead, "B's version of the key sorts above A's: newest wins"

    # B's data transaction records its own highest stamp on the scope it touched, before its COMMIT
    conn = _Conn(writing=8)
    async with CallerTransaction(conn) as transaction:
        await _epochs().touch(8, ["state:NC"], conn=conn)
        first, last = transaction.row_epoch("results"), transaction.row_epoch("results")
        assert conn.last_epochs() == [], "recorded only once the body has stamped all it will"
    assert first > stamped
    assert conn.last_epochs() == [(last, ["state:NC"])]
    assert conn.ended == [None]


async def test_a_begin_takes_the_settled_last_epoch_too() -> None:
    settled = next_epoch() + (2_000 << _LOGICAL)
    conn = _Conn(writing=9, settled_last=settled)
    async with CallerTransaction(conn):
        await _epochs().begin(conn=conn)
    assert next_epoch() > settled


async def test_a_commit_folds_the_scopes_last_epochs_and_its_own_stamps_into_the_whole_record() -> None:
    held = next_epoch()
    conn = _Conn(writing=10, pending=[{"scope": "state:NC", "epoch": 3, "pending": 10, "last_epoch": held}])
    async with CallerTransaction(conn) as transaction:
        await _epochs().touch(10, ["state:NC"], conn=conn)
        own = transaction.row_epoch("results")
        await _epochs().commit(10, (), conn=conn)
    assert (held, [WHOLE]) in conn.last_epochs()
    assert (own, [WHOLE]) in conn.last_epochs()
    assert (own, ["state:NC"]) in conn.last_epochs()


# -- s_min: every epoch record carries its commit's lowest stamp -------------------------------------------------------


@pytest.mark.parametrize("rows", [[ChangedRow(ids=["sn_NC1", "a"])], None])
def test_an_epoch_record_carries_its_commits_s_min(rows: list[ChangedRow] | None) -> None:
    s_min = next_epoch()
    body = epoch_change_body("results", rows, keys_unknown=rows is None, s_min=s_min)
    record = read_epoch_change_body(body, "results", "inc:1")
    assert record.s_min == s_min
    assert (
        read_epoch_change_body(epoch_change_body("results", rows, keys_unknown=rows is None), "results", "inc:1").s_min
        is None
    )


def test_an_s_min_that_is_no_epoch_is_refused() -> None:
    with pytest.raises(ValueError, match="s_min"):
        epoch_change_body("results", None, keys_unknown=True, s_min=-1)
