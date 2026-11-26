"""The keeper keeps the change log's marks for the log's whole window, by the window's rule, never by a switch or a TTL.

Append-only L3 chunk 7. A restore reads the versioned and stamped tables from a mark of the lost log (``table_restore``,
X = (T - M) << 15 from the newest mark at or before its floor). Chunk 5 found the marks were pruned at each ring switch,
so on a slow day the switch came after the mark the window's oldest record needed, and the restore lost it. Now a mark
goes only when it is older than the window's anchor, the newest mark before the oldest record the log holds: every
record the log holds stays vouched for, however long ago its mark was taken.
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from threetears.core.collections import change_note_keeper
from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.ddl.change_notes import MARKS_TABLE
from threetears.nats.change_log import ChangeLogState

_START = datetime(2026, 11, 25, 6, 0, tzinfo=UTC)


class _Marks:
    """the keeper's marks table, answering the anchor's keyed read and the sweep's paged delete."""

    def __init__(self, marks: list[tuple[str, int, datetime]]) -> None:
        self.rows: list[dict[str, Any]] = [
            {"id": str(uuid4()), "log_incarnation": inc, "seq": seq, "at": at} for inc, seq, at in marks
        ]
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.sent.append((query, args))
        assert MARKS_TABLE in query
        if "DELETE FROM" in query.upper():
            before = next(arg for arg in args if isinstance(arg, datetime))
            limit = re.search(r"LIMIT (\d+)", query)
            assert limit is not None
            page = int(limit[1])
            gone = sorted((row for row in self.rows if row["at"] < before), key=lambda row: row["at"])[:page]
            self.rows = [row for row in self.rows if row not in gone]
            return [{"id": row["id"]} for row in gone]
        incarnation = next(arg for arg in args if isinstance(arg, str))
        below = next(arg for arg in args if isinstance(arg, int) and not isinstance(arg, bool))
        found = [row for row in self.rows if row["log_incarnation"] == incarnation and row["seq"] < below]
        return sorted(found, key=lambda row: row["seq"], reverse=True)

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append((query, args))
        return "OK"

    def held(self) -> list[tuple[str, int]]:
        return sorted((row["log_incarnation"], row["seq"]) for row in self.rows)


def _keeper(pool: _Marks) -> ChangeNoteKeeper:
    return ChangeNoteKeeper(pool, notes_schema="platform", log=object())  # type: ignore[arg-type]


def _state(incarnation: str, first_seq: int, last_seq: int) -> ChangeLogState:
    return ChangeLogState(first_seq=first_seq, last_seq=last_seq, bytes=0, max_bytes=None, incarnation=incarnation)


async def test_a_slow_day_keeps_every_mark_the_window_needs() -> None:
    """Flips if marks of a log that still holds every record since its first mark are dropped, whatever their age:
    a slow day's window reaches back past many hours of marks, and a restore needs the oldest."""
    pool = _Marks([("inc", seq * 10, _START + timedelta(hours=seq)) for seq in range(1, 13)])
    dropped = await _keeper(pool).keep_window_marks(_state("inc", first_seq=1, last_seq=130))
    assert dropped == 0
    assert len(pool.held()) == 12


async def test_marks_before_the_window_s_anchor_go_and_the_anchor_stays() -> None:
    """Flips if a mark the window needs is dropped (the newest before the log's oldest record, or any after it), or a
    mark only older records needed is kept past the window."""
    pool = _Marks([("inc", seq, _START + timedelta(minutes=seq)) for seq in (10, 20, 30, 40, 50)])
    dropped = await _keeper(pool).keep_window_marks(_state("inc", first_seq=35, last_seq=60))
    assert dropped == 2
    # 30 is the anchor: it vouches for every record after it, the log's oldest (35) among them
    assert pool.held() == [("inc", 30), ("inc", 40), ("inc", 50)]


async def test_a_lost_log_s_marks_stay_until_the_new_log_s_window_passes_them() -> None:
    """Flips if a restore's new log, holding every record since its start, drops the lost log's marks: an unlabelled
    restore done again needs them."""
    pool = _Marks([("lost", 10, _START), ("lost", 20, _START + timedelta(minutes=1))])
    await _keeper(pool).keep_window_marks(_state("new", first_seq=1, last_seq=5))
    assert pool.held() == [("lost", 10), ("lost", 20)]
    # the new log's own marks, then its window passing its first mark: the lost log's are older than the anchor
    pool.rows += [
        {"id": str(uuid4()), "log_incarnation": "new", "seq": seq, "at": _START + timedelta(minutes=10 + seq)}
        for seq in (3, 8)
    ]
    await _keeper(pool).keep_window_marks(_state("new", first_seq=6, last_seq=20))
    assert pool.held() == [("new", 3), ("new", 8)]


def test_only_the_window_s_rule_drops_marks() -> None:
    """Flips if anything in the keeper but the window's rule drops marks: a ring switch dropping them is how a slow
    day lost the mark its restore needed (chunk 5)."""
    tree = ast.parse(inspect.getsource(change_note_keeper))
    callers = sorted(
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "drop_marks_before"
        and function.name != "drop_marks_before"
    )
    assert callers == ["keep_window_marks"]
