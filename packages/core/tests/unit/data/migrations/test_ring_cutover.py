"""The cutover's ring step takes down every ring trigger, function and table, keeps the marks, and never truncates.

Append-only L3 chunk 7 (decision 3: the ring is retired entirely). ``drop_ring`` runs once per database in the outage
window. The connection here is a stand-in for the catalog, recording every statement; the live run on Postgres and
YugabyteDB is the cutover's own proof (the replay on the new code).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from threetears.core.data.migrations.ring_cutover import RING_TABLE, RING_TABLES, drop_ring
from threetears.core.ddl.keeper_marks import MARKS_TABLE

#: the ring's row trigger and truncate trigger, as a database cut over still carries them (the ring's DDL is gone)
NOTE_TRIGGER = "threetears_change_note"
NOTE_TRUNCATE_TRIGGER = "threetears_change_note_truncate_table"


class _Conn:
    """a database holding the ring in ``platform`` and two noted tables, until the statements take it down."""

    def __init__(self) -> None:
        self.triggers = {
            ("platform", "roles", NOTE_TRIGGER),
            ("platform", "roles", NOTE_TRUNCATE_TRIGGER),
            ("agent_x", "rows", NOTE_TRIGGER),
            ("agent_x", "rows", "rows_search_vector"),
        }
        self.functions = {
            ("threetears_note_abc", "", "platform"),
            ("threetears_change_note_put", "text, text[]", "platform"),
        }
        self.tables = {("platform", table) for table in (*RING_TABLES, MARKS_TABLE)}
        self.executed: list[str] = []
        self.locked = 0

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "current_database()" in query:
            return [{"database_name": "hub"}]
        if "pg_try_advisory_lock" in query:
            self.locked += 1
            return [{"acquired": True}]
        if "pg_advisory_unlock" in query:
            self.locked -= 1
            return [{"released": True}]
        if "pg_trigger" in query:
            names = args[0]
            return [
                {"table_schema": s, "table_name": t, "trigger_name": g}
                for s, t, g in sorted(self.triggers)
                if g in names
            ]
        if "pg_proc" in query:
            return [{"name": n, "args": a} for n, a, s in sorted(self.functions) if s == args[0]]
        if "c.relname::text = $1" in query:
            return [{"schema": s} for s, t in sorted(self.tables) if t == args[0]]
        if "ANY($2::text[])" in query:
            return [{"table_name": t} for s, t in sorted(self.tables) if s == args[0] and t in args[1]]
        raise AssertionError(f"unexpected read: {query}")

    async def execute(self, query: str, *args: Any) -> str:
        assert self.locked == 1, "every drop runs under the DDL lock"
        self.executed.append(query)
        if query.startswith("DROP TRIGGER IF EXISTS"):
            name, _, target = query.removeprefix("DROP TRIGGER IF EXISTS ").partition(" ON ")
            schema, _, table = target.partition(".")
            self.triggers.discard((schema.strip('"'), table.strip('"'), name.strip('"')))
        elif query.startswith("DROP FUNCTION IF EXISTS"):
            self.functions = {f for f in self.functions if f'"{f[0]}"' not in query}
        elif query.startswith("DROP TABLE IF EXISTS"):
            self.tables = {(s, t) for s, t in self.tables if f'"{s}"."{t}"' not in query}
        return "OK"


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Conn]:
        yield self.conn


async def test_the_ring_goes_whole_and_the_marks_stay() -> None:
    """Flips if a ring trigger, note function or ring table survives the cutover, if the keeper's marks (which a restore
    from the tables still reads X from) are dropped, or if a trigger that is not the ring's is touched."""
    conn = _Conn()
    dropped = await drop_ring(_Pool(conn))
    assert conn.triggers == {("agent_x", "rows", "rows_search_vector")}
    assert conn.functions == set()
    assert conn.tables == {("platform", MARKS_TABLE)}
    assert set(dropped.tables) == {f"platform.{table}" for table in RING_TABLES}
    assert f"platform.{RING_TABLE}" in dropped.tables
    assert len(dropped.triggers) == 3


async def test_the_cutover_never_truncates_or_deletes_and_a_second_run_drops_nothing() -> None:
    """Flips if the step empties a table by TRUNCATE (it bumps YugabyteDB's catalog version cluster-wide) or by DELETE
    instead of dropping it, drops with CASCADE, or is not idempotent."""
    conn = _Conn()
    await drop_ring(_Pool(conn))
    assert all(
        each.startswith(("DROP TRIGGER IF EXISTS ", "DROP FUNCTION IF EXISTS ", "DROP TABLE IF EXISTS "))
        for each in conn.executed
    )
    assert not [each for each in conn.executed if each.upper().rstrip().endswith("CASCADE")]
    again = await drop_ring(_Pool(conn))
    assert again.triggers == again.functions == again.tables == ()
