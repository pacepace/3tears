"""Every scope a stamped write's rows name is touched before its COMMIT by 3tears itself, so a restore hears it.

A restore finds the scopes changed since X by the scope epochs table's ``last_epoch`` alone (``table_restore``). Before
this, that held only while every writer that wrote scope value ``v`` also touched ``scope_prefix + v``: a write that
touched fewer states than it wrote was missed by the restore, silently, and the scrub could not see it. Now the store's
one stamped-write path (``broker_generation.stamped_send``) reads the distinct scopes off the rows it sends, by Arrow on
the building thread, and the transaction they join raises each one's ``last_epoch`` to its highest stamp just before
its COMMIT, one statement for all of them; a versioned write on a connection no transaction holds is refused. A load
touches its file's scopes in a transaction of its own before its rows.

The L3 here is a stand-in that applies the touch statement to its scope epochs rows (``GREATEST``, as the database
does) and keeps the versions copied; the restore half reads them through ``test_table_restore``'s stand-in.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from packages.core.tests.unit.collections.test_table_restore import (
    _EPOCHS,
    _RESULTS,
    _RING,
    _SCOPE,
    _TABLE,
    _lost_log,
    _Pool,
    _restored_records,
)
from threetears.core.backends.arrow_copy import distinct_scopes
from threetears.core.backends.protocol import LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.ddl.row_epoch import (
    EPOCH_COLUMN,
    UntouchedScopesError,
    declared_versions,
    next_epoch,
    restore_declaration,
    restore_floor,
)


class _L3:
    """an asyncpg-shaped connection and pool: the scope epochs table's ``last_epoch`` by scope, raised by the touch
    statement as the database raises it, and every version copied, each with the transaction that committed it."""

    def __init__(self) -> None:
        #: scope -> last_epoch, of the scope epochs table
        self.last_epoch: dict[str, int] = {}
        #: every version copied, a dict a row
        self.rows: list[dict[str, Any]] = []
        #: each statement and its arguments, in order
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        #: what each committed transaction held, in order: "touch" and "rows"
        self.commits: list[list[str]] = []
        self._open: list[str] = []

    @asynccontextmanager
    async def transaction(self, **_: Any) -> AsyncIterator[None]:
        self._open = []
        yield
        self.commits.append(self._open)

    async def execute(self, query: str, *args: Any) -> str:
        self.statements.append((query, args))
        if "SET last_epoch" in query:
            stamp, scopes = args
            for scope in scopes:
                self.last_epoch[scope] = max(self.last_epoch.get(scope, 0), stamp)
            self._open.append("touch")
        return "INSERT 0 1"

    async def copy_records_to_table(self, table_name: str, *, records: Any, columns: Any, **_: Any) -> str:
        self.rows.extend(dict(zip(columns, record, strict=True)) for record in records)
        self._open.append("rows")
        return f"COPY {len(records)}"

    def touches(self) -> list[tuple[Any, ...]]:
        return [args for query, args in self.statements if "SET last_epoch" in query]

    @asynccontextmanager
    async def _acquire(self) -> AsyncIterator[_L3]:
        yield self

    def acquire(self) -> Any:
        return self._acquire()


def _store(l3: _L3) -> SqlL3Backend:
    store = SqlL3Backend(l3)
    store.register_schema(_TABLE, _RESULTS)
    return store


def _versions(states: list[str], epoch: int, *, runs: int = 1) -> list[dict[str, Any]]:
    return [
        {_SCOPE: state, "run_id": f"r{run}", "candidate": f"c{state}{run}", "votes": run, EPOCH_COLUMN: epoch + run}
        for state in states
        for run in range(runs)
    ]


async def test_a_writer_that_touches_no_scope_still_has_every_scope_its_rows_name_touched_and_restored() -> None:
    """Flips if a stamped write whose writer touched none of the scopes its rows name leaves any of them without its
    highest stamp as ``last_epoch`` at its COMMIT, or if a restore of a lost log then misses that scope's versions (the
    old code touched only what ``ScopeEpochs.touch`` was told, here nothing, and the restore read no scope at all)."""
    l3 = _L3()
    epoch = next_epoch()
    rows = _versions(["CA", "NV"], epoch)
    async with CallerTransaction(l3):
        # the writer "forgets": no ScopeEpochs.touch for either state
        await _store(l3).copy_many(_TABLE, rows, max_rows=100, max_bytes=1 << 20, conn=l3, append=True)

    # a restore of a lost log hears both states' versions, through last_epoch alone
    _bus, source, lost = await _lost_log()
    now = datetime.now(UTC)
    mark_at = now - timedelta(minutes=1)
    pool = _Pool(lost, now, [{"log_incarnation": lost, "seq": 0, "at": mark_at, "id": 1}])
    pool.scopes = sorted(l3.last_epoch.items())
    pool.versions = [
        (row[_SCOPE], row["run_id"], row["candidate"], row[EPOCH_COLUMN], bool(row.get("gone", False)))
        for row in l3.rows
    ]
    assert restore_floor(int(mark_at.timestamp() * 1000)) <= epoch
    await ChangeNoteKeeper(pool, notes_schema=_RING, log=source).run_pass()
    records = [record for record in await _restored_records(source) if record.table == _TABLE]
    heard = sorted(tuple(row.ids) for record in records for row in record.rows)
    assert heard == [("CA", "r0", "cCA0"), ("NV", "r0", "cNV0")], "the restore hears every scope the write wrote"
    assert l3.last_epoch == {"state:CA": epoch, "state:NV": epoch}, "every scope the rows name, at the highest stamp"
    assert l3.commits == [["rows", "touch"]], "touched in the rows' own transaction, before its COMMIT"


async def test_a_versioned_write_no_transaction_holds_is_refused_before_anything_is_sent() -> None:
    """Flips if a stamped write of a versioned table is sent on a connection no transaction holds to touch its scopes
    in before its COMMIT (the old code sent it, and no scope was ever touched)."""
    l3 = _L3()
    with pytest.raises(UntouchedScopesError, match="joins no transaction"):
        await _store(l3).copy_many(
            _TABLE, _versions(["CA"], next_epoch()), max_rows=100, max_bytes=1 << 20, conn=l3, append=True
        )
    assert l3.rows == [] and l3.statements == []


async def test_a_5000_row_write_of_50_scopes_touches_each_scope_once_in_one_statement() -> None:
    """Flips if a transaction's touch costs more than one statement and one index entry a scope: 5,000 rows of 50 states
    written in five pages, and ScopeEpochs.touch naming some of the same states, must send one touch naming the 50."""
    l3 = _L3()
    states = [f"S{index:02d}" for index in range(50)]
    epoch = next_epoch()
    rows = _versions(states, epoch, runs=100)
    assert len(rows) == 5000
    async with CallerTransaction(l3) as transaction:
        transaction.touch_scopes(_EPOCHS, [f"state:{state}" for state in states[:10]])
        for start in range(0, len(rows), 1000):
            await _store(l3).copy_many(
                _TABLE, rows[start : start + 1000], max_rows=1000, max_bytes=1 << 24, conn=l3, append=True
            )
    [(stamp, scopes)] = l3.touches()
    assert stamp == epoch + 99, "the transaction's highest stamp"
    assert scopes == sorted(f"state:{state}" for state in states)


def test_the_scopes_of_5000_rows_are_found_by_arrow_in_well_under_a_millisecond_a_page() -> None:
    """Flips if finding a page's distinct scopes stops being an Arrow pass (a Python step a row would be ~10x this)."""
    values = pa.array([f"S{index % 50:02d}" for index in range(5000)])
    distinct_scopes(values, "state:")
    started = time.perf_counter()
    for _ in range(20):
        found = distinct_scopes(values, "state:")
    each_ms = (time.perf_counter() - started) * 1000 / 20
    assert len(found) == 50 and found[0] == "state:S00"
    assert each_ms < 5.0, f"{each_ms:.3f} ms to find 50 scopes in 5,000 rows"


async def test_a_direct_load_touches_its_files_scopes_in_a_transaction_before_its_rows(tmp_path: Path) -> None:
    """Flips if a load of a versioned table commits rows before the scopes they name are touched with the load's stamp
    (the old code touched none: a restore missed every loaded state)."""
    l3 = _L3()
    epoch = next_epoch()
    sink = io.BytesIO()
    rows = [{_SCOPE: state, "run_id": "r1", "candidate": f"c{state}", "votes": 1} for state in ("TX", "AZ", "TX")]
    rows[2]["candidate"] = "cTX2"
    pq.write_table(pa.Table.from_pylist(rows), sink)
    path = tmp_path / "file.parquet"
    path.write_bytes(sink.getvalue())
    await _store(l3).load_files(
        _TABLE,
        [LoadFile(url=path.as_uri(), rows=3)],
        columns=[_SCOPE, "run_id", "candidate", "votes"],
        constants={EPOCH_COLUMN: epoch},
        rows_per_transaction=2,
        timeout_ms=10_000,
    )
    assert l3.last_epoch == {"state:AZ": epoch, "state:TX": epoch}
    assert l3.commits[0] == ["touch"] and all(commit == ["rows"] for commit in l3.commits[1:]), l3.commits


def test_a_versioned_tables_declaration_reads_back_from_its_index_comment() -> None:
    """Flips if what the hub reads off a versioned table's (scope, epoch) index comment is not the table's own scope
    column, scope epochs table and prefix (the broker touches a load's scopes by it), or a comment that is none reads
    as one."""
    comment = restore_declaration(_RESULTS.row_epoch, tuple(_RESULTS.pk_columns), (), ("schema", None))
    assert declared_versions(comment) == _RESULTS.row_epoch
    assert declared_versions(None) is None and declared_versions('{"other": 1}') is None
    assert declared_versions("not json") is None
