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
from threetears.core.collections.write_marks import WriteMarkReconciler

_EPOCH = datetime(2026, 11, 11, tzinfo=UTC)

#: a load of two states' rows, as ``note_reach_text(prefixes=...)`` writes it
_LOAD = json.dumps({"prefixes": [{"state": "PA"}, {"state": "OH"}]}, separators=(",", ":"))
_RESET = json.dumps({"reason": "reset"}, separators=(",", ":"))


#: ``_noted_rows_sql``'s select list, in order
_FIELDS = ("switch_no", "previous_switched_at", "reach", "noted", "key_text", "keys", "columns", "tenant")


class _Record(tuple[Any, ...]):
    """an asyncpg record: read by position or by name."""

    def __getitem__(self, item: Any) -> Any:  # type: ignore[override]
        return super().__getitem__(_FIELDS.index(item) if isinstance(item, str) else item)


def _note(key_text: str | None, keys: list[str] | None, columns: Any, tenant: str | None) -> _Record:
    """one row of ``_noted_rows_sql``: the ring (no switch yet), then a note."""
    return _Record((0, _EPOCH, _EPOCH, True, key_text, keys, columns, tenant))


class _Notes:
    """a pool answering the notes statement with the rows given, and the reconciler's marks page once."""

    def __init__(self, notes: list[_Record], table: str = "results") -> None:
        self.notes = notes
        self.marks = {1: table}

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if "current_ring" in query:
            return list(self.notes)
        if "count(*)" in query:
            return [{"n": len(self.marks), "age": 3.0 if self.marks else None}]
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
    answer = await noted_rows_since(_Notes(notes), notes_schema="platform", table="results", since=_EPOCH, cap=100)
    assert isinstance(answer, NotedRows), answer
    return answer


async def test_a_load_s_whole_note_keeps_its_prefixes_and_covers_no_row_beside_them() -> None:
    """Flips if a load's whole note comes back as the plain whole table (its prefixes dropped), or if a row its
    tenant wrote beside the load is no longer named (a load names only its prefixes)."""
    noted = await _noted([_note(None, None, _LOAD, "s1"), _note("x", ["x"], None, "s1")])
    assert noted.whole == (
        NotedChange(table="results", tenant="s1", whole_table=True, prefixes=({"state": "OH"}, {"state": "PA"})),
    )
    assert [row.ids for row in noted.rows] == [["x"]]
    assert noted.whole_table and noted.whole_tenants == ("s1",)


async def test_a_reset_s_whole_note_keeps_its_reason_and_covers_its_tenant_s_rows() -> None:
    """Flips if an operator's reset comes back without its reason, or a row it covers is still named."""
    noted = await _noted([_note(None, None, _RESET, None), _note("x", ["x"], None, "s1")])
    assert noted.whole == (NotedChange(table="results", tenant=None, whole_table=True, reason="reset"),)
    assert noted.rows == ()


async def test_a_whole_note_naming_no_reach_is_the_plain_whole_table() -> None:
    """Flips if a bare truncate's note gains a reach, or stops covering its tenant's rows."""
    noted = await _noted([_note(None, None, None, "s1"), _note("x", ["x"], None, "s1"), _note("y", ["y"], None, "s2")])
    assert noted.whole == (NotedChange(table="results", tenant="s1", whole_table=True),)
    assert [(row.ids, row.tenant) for row in noted.rows] == [(["y"], "s2")]


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
            generation_source=source,  # type: ignore[arg-type]
            marks_schema="platform",
            tables=lambda: ("results",),
        )
        done = await reconciler.run_pass()
        assert done.applied == 1, done
        (change,) = source.changes
        assert change["rows"] is None and not change["keys_unknown"], change
        assert (change["prefixes"], change["reason"]) == (prefixes, reason), change
