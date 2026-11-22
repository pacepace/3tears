"""A whole-table note keeps its reach through the write-mark reconciler: a load's key prefixes, an operator's reset.

A file load (or a clear) marks its table whole and names the key prefixes it reached; a reset names its reason. The
note stores that reach (``NOTE_REACH_SETTING``), and the reconciler that advances a marked table from its notes must
record the same reach, as a restore does: dropped, the load's leftover mark becomes a legacy whole-table change and
every follower reads the table whole."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from threetears.core.collections.change_notes import NotedChange, NotedRows, noted_rows_since
from threetears.core.collections.generation import tables_written_l2_first
from threetears.core.collections.write_marks import WriteMarkReconciler

_EPOCH = datetime(2026, 11, 11, tzinfo=UTC)

#: the noted table: a name no collection class in the process claims. The reconciler passes over a table some class
#: defined in the process writes L2-first (``tables_written_l2_first``, a process-wide registry), and the epoch
#: package's L2-first tests define such a class on "results": run in one batch after them, a "results" mark here was
#: deleted unadvanced and the reconciler test failed on the batch's order alone
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
    """a pool answering the notes statement with the rows given, and the reconciler's marks page once."""

    def __init__(self, notes: list[_Record], table: str = _TABLE) -> None:
        self.notes = notes
        self.marks = {1: table}

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if "current_ring" in query:
            return list(self.notes)
        if "count(*)" in query:
            # the gauges' bounded count (keyed.count_at_most): the marks, at most the bound it asks past
            return [{"held": min(len(self.marks), int(args[-1]))}]
        if "extract(epoch" in query:
            # the gauges' oldest mark, the lowest id (keyed.keyset_page, one row)
            return [{"id": min(self.marks), "age": 3.0 if self.marks else None}] if self.marks else []
        if "WHERE id > $2 ORDER BY id LIMIT $1" in query:
            return [
                {"id": key, "table_name": value, "date_created": _EPOCH + timedelta(seconds=1)}
                for key, value in sorted(self.marks.items())
                if key > int(args[1])
            ][: int(args[0])]
        if "information_schema.tables" in query:
            return []
        return []

    async def execute(self, query: str, *args: Any) -> str:
        assert query.startswith("DELETE"), query
        for key in args[0]:
            self.marks.pop(key, None)
        return f"DELETE {len(args[0])}"


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


async def test_a_whole_note_older_than_the_marked_writes_is_not_theirs() -> None:
    """Flips if a whole-table note stamped before the oldest marked write began (an earlier write's, whose advance
    was heard) comes back as one of the marked writes' changes: the reconciler then re-recorded it, and every
    follower dropped the table and read it whole again (2026-11-22 replay: one reset, re-announced three times)."""
    old_reset = _note(None, None, _RESET, None, noted=_EPOCH - timedelta(minutes=3))
    noted = await _noted([old_reset, _note("x", ["x"], None, "s1")])
    assert noted.whole == ()
    assert [row.ids for row in noted.rows] == [["x"]]


async def test_the_reconciler_records_a_load_s_mark_with_its_prefixes_and_a_reset_with_its_reason() -> None:
    """Flips if the write-mark reconciler records a load's leftover mark as the legacy whole table (followers would
    read it whole) or a reset without its reason."""
    for notes, prefixes, reason in (
        ([_note(None, None, _LOAD, "s1")], [{"state": "OH"}, {"state": "PA"}], None),
        ([_note(None, None, _RESET, None)], None, "reset"),
    ):
        pool, source = _Notes(notes), _Recorder()
        reconciler = WriteMarkReconciler(
            pool,  # type: ignore[arg-type]
            generation_source=source,
            marks_schema="platform",
            tables=lambda: (_TABLE,),
            settle_seconds=0.0,
        )
        assert _TABLE not in tables_written_l2_first(), "a class in this process claims the test's table L2-first"
        done = await reconciler.run_pass()
        assert done.applied == 1, done
        (change,) = source.changes
        assert change["rows"] is None and not change["keys_unknown"], change
        assert (change["prefixes"], change["reason"]) == (prefixes, reason), change
