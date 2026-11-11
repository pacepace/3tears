"""unit: every 3tears function that runs DDL on the platform's database takes the database's DDL lock in its own body.

On YugabyteDB two index builds in one database wait for each other until the catalog-version wait times out and both
indexes are left invalid (:mod:`threetears.core.ddl.ddl_lock`). A function that drops and builds indexes, or installs
a table and its indexes, therefore holds the lock itself, whoever calls it: the hub's start, a migration, an installer.
The lock is re-entrant on its session, so a caller that already holds it loses nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.ddl.index_health import rebuild_unusable_indexes

_INDEX_DEF = 'CREATE INDEX "ring_key" ON ring.notes USING lsm (k)'


class _Conn:
    """one asyncpg-shaped connection: answers the DDL lock, an unusable index (until it is built again), and
    records every statement in order."""

    def __init__(self, *, unusable: bool) -> None:
        self.statements: list[str] = []
        self._unusable = unusable

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(sql)
        if "current_database" in sql:
            return [{"database_name": "aibots"}]
        if "pg_try_advisory_lock" in sql:
            return [{"acquired": True}]
        if "pg_advisory_unlock" in sql:
            return [{"released": True}]
        if "indisvalid" in sql and self._unusable:
            return [
                {
                    "schema_name": "ring",
                    "table_name": "notes",
                    "index_name": "ring_key",
                    "is_unique": False,
                    "definition": _INDEX_DEF,
                    "predicate": None,
                    "key_terms": ["k"],
                }
            ]
        return []

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append(sql)
        if sql == _INDEX_DEF:
            self._unusable = False
        return "OK"


def _ddl(statement: str) -> bool:
    return statement.lstrip().upper().startswith(("CREATE ", "DROP ", "ALTER "))


def _locked_around_every_ddl(statements: list[str]) -> bool:
    """whether a lock was taken before the first DDL statement and given back only after the last."""
    ddl = [n for n, s in enumerate(statements) if _ddl(s)]
    locks = [n for n, s in enumerate(statements) if "pg_try_advisory_lock" in s]
    unlocks = [n for n, s in enumerate(statements) if "pg_advisory_unlock" in s]
    return bool(ddl) and bool(locks) and bool(unlocks) and locks[0] < ddl[0] and unlocks[-1] > ddl[-1]


async def test_rebuilding_an_unusable_index_holds_the_database_ddl_lock_around_its_drop_and_build() -> None:
    """Flips if rebuild_unusable_indexes drops or builds an index without the database's DDL lock held."""
    conn = _Conn(unusable=True)
    rebuilt = await rebuild_unusable_indexes(conn, ["ring"])
    assert [index.name for index in rebuilt] == ["ring_key"]
    assert 'DROP INDEX IF EXISTS "ring"."ring_key"' in conn.statements
    assert _locked_around_every_ddl(conn.statements), conn.statements


async def test_a_start_with_every_index_usable_takes_no_ddl_lock() -> None:
    """Flips if a repair with nothing to build takes the database's DDL lock (every start would queue on it)."""
    conn = _Conn(unusable=False)
    assert await rebuild_unusable_indexes(conn, ["ring"]) == []
    assert not any("advisory" in s for s in conn.statements), conn.statements


_DECLARED = 'CREATE INDEX IF NOT EXISTS "ring_key" ON ring.notes (k, noted_at)'


class _UniqueLeftover(_Conn):
    """the same connection, whose unusable ``ring_key`` a failed build of a UNIQUE index left under the ring's name:
    usable again only once the ring's declared statement runs."""

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        rows = await super().fetch(sql, *params)
        if "indisvalid" in sql:
            return [
                {**row, "is_unique": True, "definition": "CREATE UNIQUE INDEX ring_key ON ring.notes (k)"}
                for row in rows
            ]
        if "HAVING count(*) > 1" in sql:
            return [{"k0": "lsf_runs", "n": 50}]
        return rows

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append(sql)
        if sql == _DECLARED:
            self._unusable = False
        return "OK"


async def test_an_installer_s_unusable_index_is_built_from_its_declared_statement() -> None:
    """Flips if a declared index name is built again from what a failed build left under it (a UNIQUE index the
    table's duplicates refuse) rather than from the installer's own statement."""
    conn = _UniqueLeftover(unusable=True)
    rebuilt = await rebuild_unusable_indexes(conn, ["ring"], names=["ring_key"], definitions={"ring_key": _DECLARED})
    assert [index.name for index in rebuilt] == ["ring_key"]
    assert _DECLARED in conn.statements
    assert not any(s.startswith("CREATE UNIQUE") for s in conn.statements), conn.statements
    assert not any("HAVING count(*) > 1" in s for s in conn.statements), conn.statements


class _Pool:
    """an asyncpg-shaped pool handing out one connection."""

    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def _acquire(self) -> AsyncIterator[_Conn]:
        yield self._conn

    def acquire(self) -> Any:
        return self._acquire()


async def test_installing_the_change_note_ring_holds_the_database_ddl_lock_around_its_ddl() -> None:
    """Flips if the change-note ring's install runs its CREATE statements without the database's DDL lock held
    (the lock is taken in the install's own body, so no caller of it can run the ring's DDL unlocked)."""
    conn = _Conn(unusable=False)
    keeper = ChangeNoteKeeper(_Pool(conn), notes_schema="ring", log=None)  # type: ignore[arg-type]
    await keeper.install()
    assert any(_ddl(s) for s in conn.statements)
    assert _locked_around_every_ddl(conn.statements), conn.statements
