"""A whole-table note keeps its reach through a lost write's recovery: a load's key prefixes, an operator's reset.

A file load (or a clear) marks its table whole and names the key prefixes it reached; a reset names its reason. The
note stores that reach (``NOTE_REACH_SETTING``), and a recovery that advances a table from its window's notes must
record the same reach, as a restore does: dropped, the load would become a legacy whole-table change and every
follower would read the table whole."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from threetears.core.collections.change_notes import NotedChange, NotedRows, noted_rows_since

_EPOCH = datetime(2026, 11, 11, tzinfo=UTC)

#: the noted table: a name no collection class in the process claims. A recovery passes over a table some class
#: defined in the process writes L2-first (``written_l2_first``, a process-wide registry), and the epoch
#: package's L2-first tests define such a class on "results": run in one batch after them, the recovery test here would
#: fail on the batch's order alone
_TABLE = "noted_reach_results"

#: a load of two states' rows, as ``note_reach_text(prefixes=...)`` writes it
_LOAD = json.dumps({"prefixes": [{"state": "PA"}, {"state": "OH"}]}, separators=(",", ":"))
_RESET = json.dumps({"reason": "reset"}, separators=(",", ":"))


#: ``noted_rows_since``'s select list (``read_row_beside``: the ring row's columns, then the notes'), in order
_FIELDS = (
    "current_ring",
    "switch_no",
    "previous_switched_at",
    "wait_seconds",
    "writes_previous_until",
    "noted",
    "ring_row",
    "key_text",
    "keys",
    "columns",
    "tenant",
)


class _Record(tuple[Any, ...]):
    """an asyncpg record: read by position or by name."""

    def __getitem__(self, item: Any) -> Any:
        return super().__getitem__(_FIELDS.index(item) if isinstance(item, str) else item)


def _note(
    key_text: str | None, keys: list[str] | None, columns: Any, tenant: str | None, *, noted: datetime = _EPOCH
) -> _Record:
    """one row of ``noted_rows_since``'s statement: the ring (no switch yet; its columns answered on the first note
    alone, as ``read_row_beside(first=)`` answers them), then a note stamped ``noted`` (microseconds since the epoch,
    as the statement answers it)."""
    micros = (noted - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)
    return _Record((0, 0, _EPOCH, 60.0, _EPOCH, micros, True, key_text, keys, columns, tenant))


class _Notes:
    """a pool answering the notes statement with the rows given, recording what it was asked."""

    def __init__(self, notes: list[_Record]) -> None:
        self.notes = notes
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.asked.append((query, args))
        if "current_ring" in query:
            return list(self.notes)
        return []

    async def execute(self, query: str, *args: Any) -> str:
        raise AssertionError(f"a recovery writes nothing to L3: {query}")


class _Recorder:
    """a source that records each change it is handed, with its reach."""

    def __init__(self) -> None:
        self.changes: list[dict[str, Any]] = []

    async def advance(self, table_name: str) -> str:
        raise AssertionError("a recording source is advanced with its change")

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Any,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        partition: Any = None,
        prefixes: Any = None,
        reason: str | None = None,
    ) -> str:
        self.changes.append(
            {
                "table": table_name,
                "rows": None if rows is None else sorted(tuple(row.ids) for row in rows),
                "keys_unknown": keys_unknown,
                "tenant": tenant,
                "prefixes": prefixes,
                "reason": reason,
            }
        )
        return f"inc:{len(self.changes)}"


async def _noted(notes: list[_Record]) -> NotedRows:
    answer = await noted_rows_since(_Notes(notes), notes_schema="platform", table=_TABLE, since=_EPOCH, cap=100)
    assert isinstance(answer, NotedRows), answer
    return answer


async def test_a_load_s_whole_note_keeps_its_prefixes_and_covers_no_row_beside_them() -> None:
    """Flips if a load's whole note comes back as the plain whole table (its prefixes dropped), or if a row its
    tenant wrote beside the load is no longer named (a load names only its prefixes)."""
    noted = await _noted([_note(None, None, _LOAD, "s1"), _note("x", ["x"], None, "s1")])
    assert noted.whole == (
        NotedChange(table=_TABLE, tenant="s1", whole_table=True, prefixes=({"state": "OH"}, {"state": "PA"})),
    )
    assert [row.ids for row in noted.rows] == [["x"]]
    assert noted.whole_table and noted.whole_tenants == ("s1",)


async def test_a_reset_s_whole_note_keeps_its_reason_and_covers_its_tenant_s_rows() -> None:
    """Flips if an operator's reset comes back without its reason, or a row it covers is still named."""
    noted = await _noted([_note(None, None, _RESET, None), _note("x", ["x"], None, "s1")])
    assert noted.whole == (NotedChange(table=_TABLE, tenant=None, whole_table=True, reason="reset"),)
    assert noted.rows == ()


async def test_a_whole_note_naming_no_reach_is_the_plain_whole_table() -> None:
    """Flips if a bare truncate's note gains a reach, or stops covering its tenant's rows."""
    noted = await _noted([_note(None, None, None, "s1"), _note("x", ["x"], None, "s1"), _note("y", ["y"], None, "s2")])
    assert noted.whole == (NotedChange(table=_TABLE, tenant="s1", whole_table=True),)
    assert [(row.ids, row.tenant) for row in noted.rows] == [(["y"], "s2")]


async def test_a_whole_note_is_read_only_from_the_window_s_own_start() -> None:
    """Flips if the statement reads a whole-table note from before the window (less the clocks' skew) or after its end:
    an earlier write's whole note, whose advance was heard, was re-recorded and every follower dropped the table and
    read it whole again (2026-11-22 replay: one reset, re-announced three times). The bound is the statement's: the
    answered row's ``noted`` column is TRUE, never a time (6a31e38aa compared it with a time and dropped every whole
    note). Live: ``test_lost_writes_live.py``."""
    until = _EPOCH + timedelta(minutes=1)
    pool = _Notes([_note("x", ["x"], None, "s1")])
    await noted_rows_since(pool, notes_schema="platform", table=_TABLE, since=_EPOCH, until=until, cap=100)
    ((query, args),) = pool.asked
    assert "t.key_text IS NOT NULL OR t.noted_at >=" in query
    assert _EPOCH - timedelta(seconds=5) in args
    assert until in args
