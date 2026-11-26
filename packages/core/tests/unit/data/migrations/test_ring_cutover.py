"""The cutover's ring step takes down every ring trigger, function and table, keeps the marks, and never truncates.

Append-only L3 chunk 7 (decision 3: the ring is retired entirely). ``drop_ring_on`` runs once per database, on the
boot migration's session. The connection here is a stand-in for the catalog, recording every statement, with the DDL lock as a session
lock: re-entrant on the session holding it, refused to any other. The live run on Postgres is
``tests/integration/migrations/test_drop_ring_on_held_session_against_postgres.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from threetears.core.data.migrations.ring_cutover import RING_TABLE, RING_TABLES, drop_ring_on
from threetears.core.data.store import DataStore
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.errors import DdlLockTimeoutError
from threetears.core.ddl.keeper_marks import MARKS_TABLE
from threetears.core.ddl.session import ConnectionSession

#: the ring's row trigger and truncate trigger, as a database cut over still carries them (the ring's DDL is gone)
NOTE_TRIGGER = "threetears_change_note"
NOTE_TRUNCATE_TRIGGER = "threetears_change_note_truncate_table"


#: a step that waits for the lock would wait forever here; the test's own bound, far past one that never waits
_NEVER_WAITS_SECONDS = 5.0


@dataclass
class _Lock:
    """the database's DDL lock, as a session lock: the session holding it and how many holds it has."""

    holder: object | None = None
    holds: int = 0


class _Conn:
    """a database holding the ring in ``platform`` and two noted tables, until the statements take it down.

    :param lock: the database's DDL lock, shared by every connection of one database
    """

    def __init__(self, lock: _Lock | None = None) -> None:
        self.lock = lock if lock is not None else _Lock()
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

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "current_database()" in query:
            return [{"database_name": "hub"}]
        if "pg_try_advisory_lock" in query:
            if self.lock.holder not in (None, self):
                return [{"acquired": False}]
            self.lock.holder, self.lock.holds = self, self.lock.holds + 1
            return [{"acquired": True}]
        if "pg_advisory_unlock" in query:
            if self.lock.holder is not self:
                return [{"released": False}]
            self.lock.holds -= 1
            if self.lock.holds == 0:
                self.lock.holder = None
            return [{"released": True}]
        if "FROM pg_namespace n" in query:
            ring, names = args
            held = {s for s, t in self.tables if t == ring} | {s for s, _, g in self.triggers if g in names}
            return [{"schema": s} for s in sorted(held)]
        if "pg_trigger" in query:
            schema, names = args
            return [
                {"table_name": t, "trigger_name": g} for s, t, g in sorted(self.triggers) if s == schema and g in names
            ]
        if "pg_proc" in query:
            return [{"name": n, "args": a} for n, a, s in sorted(self.functions) if s == args[0]]
        if "ANY($2::text[])" in query:
            return [{"table_name": t} for s, t in sorted(self.tables) if s == args[0] and t in args[1]]
        raise AssertionError(f"unexpected read: {query}")

    async def execute(self, query: str, *args: Any) -> str:
        assert self.lock.holder is self, "every drop runs under the DDL lock, on the session holding it"
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


async def test_the_ring_goes_whole_and_the_marks_stay() -> None:
    """Flips if a ring trigger, note function or ring table survives the cutover, if the keeper's marks (which a restore
    from the tables still reads X from) are dropped, or if a trigger that is not the ring's is touched."""
    conn = _Conn()
    dropped = await drop_ring_on(ConnectionSession(conn))
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
    await drop_ring_on(ConnectionSession(conn))
    assert all(
        each.startswith(("DROP TRIGGER IF EXISTS ", "DROP FUNCTION IF EXISTS ", "DROP TABLE IF EXISTS "))
        for each in conn.executed
    )
    assert not [each for each in conn.executed if each.upper().rstrip().endswith("CASCADE")]
    again = await drop_ring_on(ConnectionSession(conn))
    assert again.triggers == again.functions == again.tables == ()


async def test_on_a_session_holding_the_lock_the_step_takes_it_again_at_once_and_drops_the_ring() -> None:
    """Flips if the held-session form takes the lock on any connection but the one it is handed (the boot migration
    would wait on its own run forever), waits for the lock its session already holds, or gives back the caller's hold
    along with its own."""
    conn = _Conn()
    session = ConnectionSession(conn)
    async with database_ddl_lock(session):
        dropped = await asyncio.wait_for(drop_ring_on(session), _NEVER_WAITS_SECONDS)
        assert (conn.lock.holder, conn.lock.holds) == (conn, 1), "the caller still holds its own hold"
    assert conn.lock.holder is None
    assert conn.tables == {("platform", MARKS_TABLE)}
    assert conn.functions == set()
    assert len(dropped.triggers) == 3


async def test_a_migration_body_s_store_is_used_as_it_is_under_the_run_s_lock() -> None:
    """Flips if a store holding the run's lock (a migration body's) is not handed the drops on its own session, or the
    step asks for the lock again past it; the hub's v193 passes its store."""
    conn = _Conn()
    session = ConnectionSession(conn)
    async with database_ddl_lock(session):
        store = DataStore.over_session(session, holds_ddl_lock=True)
        dropped = await asyncio.wait_for(drop_ring_on(store, policy=DdlLockPolicy(max_wait=0)), _NEVER_WAITS_SECONDS)
        assert conn.lock.holds == 1
    assert set(dropped.tables) == {f"platform.{table}" for table in RING_TABLES}
    assert conn.tables == {("platform", MARKS_TABLE)}


async def test_a_second_connection_of_the_database_is_refused_the_held_lock() -> None:
    """The premise the held-session form exists for: while one session holds the DDL lock, the step on another
    connection of the database cannot get it. Flips if the stand-in stops modelling a session lock (and the test above
    stops proving anything)."""
    lock = _Lock()
    held, other = _Conn(lock), _Conn(lock)
    async with database_ddl_lock(ConnectionSession(held)):
        with pytest.raises(DdlLockTimeoutError):
            await drop_ring_on(ConnectionSession(other), policy=DdlLockPolicy(poll_interval=0.01, max_wait=0.05))
    assert other.executed == []
