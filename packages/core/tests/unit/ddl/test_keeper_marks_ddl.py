"""the log keeper's DDL: its marks table alone, and no change-note ring.

The change-note ring retired (append-only L3, ``design-append-only-l3.md`` decision 3, Pace 2026-11-25). What the
keeper installs, and what a migration run does at its end, must create no ring table, no ring row, no note function and
no trigger: a ring piece left behind is the per-row path back (1.58 ms a row against 0.43 without, 2026-11-25).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.core.collections.log_keeper import LogKeeper
from threetears.core.data.migrations import epoch_refusals
from threetears.core.ddl.index_health import ddl_body
from threetears.core.ddl.keeper_marks import MARKS_TABLE, keeper_marks_ddl

#: what any piece of the ring is named by: its note tables, its ring row, its functions and triggers
_RING_NAMES = ("threetears_change_notes_", "threetears_change_note_ring", "threetears_note_", 'threetears_change_note"')


#: a statement creating a trigger or a function
_MAKES_TRIGGER_OR_FUNCTION = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:CONSTRAINT\s+)?(?:TRIGGER|FUNCTION)\b", re.I)


class _Conn:
    """the install's one connection: every statement recorded, every catalog read answering nothing."""

    def __init__(self, sent: list[str]) -> None:
        self._sent = sent

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self._sent.append(query)
        # the DDL lock's own statements: the database, the lock granted, then released
        if "current_database()" in query:
            return [{"database_name": "platform"}]
        if "pg_try_advisory_lock" in query:
            return [{"acquired": True}]
        if "pg_advisory_unlock" in query:
            return [{"released": True}]
        return []

    async def fetchval(self, query: str, *args: Any) -> Any:
        self._sent.append(query)
        return True

    async def execute(self, query: str, *args: Any) -> str:
        self._sent.append(query)
        return "OK"

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class _Pool:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append(query)
        return []

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append(query)
        return "OK"

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Conn]:
        yield _Conn(self.sent)


def test_the_marks_install_rebuilds_no_unique_index() -> None:
    """the proof of the keeper's install's declared ``schema_change`` call into ``rebuild_unusable_indexes``: it hands
    the repair the marks' own statements as the definitions, and a declared statement that is not UNIQUE needs no
    duplicate-key check, the only read of a table's rows the repair makes. Every marks index is a plain
    ``CREATE INDEX``, so the install never reads the marks whole; a unique marks index would flip this."""
    bodies = [ddl_body(statement).upper() for statement in keeper_marks_ddl("platform")]
    indexes = [body for body in bodies if body.startswith(("CREATE INDEX", "CREATE UNIQUE INDEX"))]
    assert len(indexes) == 2, "the marks declare their position and time indexes"
    assert not [body for body in indexes if body.startswith("CREATE UNIQUE ")]


async def test_no_ring_table_or_trigger_exists_after_the_keepers_ddl() -> None:
    """Flips if the keeper's install creates any piece of the change-note ring (a note table, the ring row, a note
    function, a trigger), or creates anything but the marks table and its indexes."""
    pool = _Pool()
    await LogKeeper(pool, schema="platform", log=object()).install()  # type: ignore[arg-type]
    created = [ddl_body(query).upper() for query in pool.sent if ddl_body(query).upper().startswith("CREATE")]
    assert created and all(MARKS_TABLE.upper() in statement for statement in created), created
    ring = [query for query in pool.sent if any(name in query for name in _RING_NAMES)]
    assert ring == [], ring
    made = [query for query in pool.sent if _MAKES_TRIGGER_OR_FUNCTION.search(query)]
    assert made == [], made


def test_no_migration_run_gives_a_table_a_trigger() -> None:
    """Flips if the migration run's end gains a step that gives a table a change-note trigger again: its module holds the
    three refusals alone (a per-row trigger on a versioned table; a switched-on table with no epoch; a moveable stamped
    table whose schema holds no moves table)."""
    assert sorted(epoch_refusals.__all__) == [
        "refuse_moveable_without_moves_here",
        "refuse_row_triggers_here",
        "switched_tables_without_epoch",
    ]
    for gone in ("give_noted_tables_their_note_trigger", "switched_tables_without_notes"):
        assert not hasattr(epoch_refusals, gone), gone


@pytest.mark.parametrize(
    "module",
    [
        "threetears.core.ddl.change_notes",
        "threetears.core.collections.change_note_keeper",
        "threetears.core.utils.change_note_slot",
        "threetears.core.testing.note_ring",
        "threetears.core.data.migrations.change_notes",
    ],
)
def test_the_ring_s_modules_are_gone(module: str) -> None:
    """Flips if a ring module comes back: its DDL, its keeper, its writers' cached slot, its test helper, its migration
    step."""
    with pytest.raises(ModuleNotFoundError):
        __import__(module)
