"""live: a table switched off loses what the runner's two hooks gave it, and nothing else does.

Against a real database (the session ``db_container``). A schema holds the write-mark queue (coordination's
migrations at platform scope) and the change-note ring, as the platform schema does; a table there is switched on by
the two hooks, then off by :func:`switch_off`: its write-mark trigger, its note triggers and its note function go,
each ``IF EXISTS``, so the step runs twice and on a dropped table. A same-named table in another schema keeps its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.collections.generation import NotedTable
from threetears.core.coordination.migrations import register as register_coordination
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.migrations.change_notes import give_noted_tables_their_note_trigger
from threetears.core.data.migrations.session import ConnectionSession
from threetears.core.data.migrations.write_generation_switch import switch_off, write_generation_triggers_on
from threetears.core.data.migrations.write_marks import give_switched_tables_their_trigger
from threetears.core.ddl.change_notes import NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER, change_notes_ddl, note_function_name
from threetears.core.ddl.write_marks import WRITE_MARK_TRIGGER
from threetears.core.tenancy import TENANT_BY_SCHEMA

pytestmark = pytest.mark.integration

_TABLE = "switched_widgets"
_ON = {WRITE_MARK_TRIGGER, NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER}


async def _schema(conn: asyncpg.Connection, schema: str) -> ConnectionSession:
    """``schema`` holding the queue, the ring and the table, switched on by the two hooks, bound to the session."""
    await conn.execute(f"CREATE SCHEMA {schema}")
    await conn.execute(f"SET search_path TO {schema}")
    runner = MigrationRunner()
    register_coordination(runner)
    session = ConnectionSession(conn)
    await runner.apply_for_platform_schema(session)
    for statement in change_notes_ddl(schema):
        await conn.execute(statement)
    await conn.execute(f"CREATE TABLE {schema}.{_TABLE} (id TEXT PRIMARY KEY, n INTEGER)")
    await give_switched_tables_their_trigger(session, [_TABLE])
    await give_noted_tables_their_note_trigger(session, [NotedTable(_TABLE, keys=("id",), tenancy=TENANT_BY_SCHEMA)])
    return session


async def _functions(conn: asyncpg.Connection, schema: str) -> set[str]:
    rows = await conn.fetch(
        "SELECT p.proname AS name FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = $1 AND p.proname = $2",
        schema,
        note_function_name(schema, _TABLE),
    )
    return {str(row["name"]) for row in rows}


@pytest.fixture
async def two(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, asyncpg.Connection, str, str]]:
    """two schemas, each with the table switched on, each on a connection bound to it."""
    suffix = uuid.uuid4().hex[:8]
    names = (f"switch_a_{suffix}", f"switch_b_{suffix}")
    first, second = await asyncpg.connect(db_container), await asyncpg.connect(db_container)
    try:
        await _schema(first, names[0])
        await _schema(second, names[1])
        yield first, second, names[0], names[1]
    finally:
        for name in names:
            await first.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
        await first.close()
        await second.close()


async def test_switching_off_takes_the_triggers_and_the_note_function_away_and_only_there(
    two: tuple[asyncpg.Connection, asyncpg.Connection, str, str],
) -> None:
    first, second, schema, other = two
    session = ConnectionSession(first)
    assert await write_generation_triggers_on(session, _TABLE) == _ON
    assert await _functions(first, schema) == {note_function_name(schema, _TABLE)}

    await switch_off(session, _TABLE)

    assert await write_generation_triggers_on(session, _TABLE) == frozenset()
    assert await _functions(first, schema) == set()
    # a write to it leaves no mark and no note
    await first.execute(f"INSERT INTO {schema}.{_TABLE} VALUES ('a', 1)")
    assert await first.fetchval(f"SELECT count(*) FROM {schema}.collection_write_marks") == 0
    # the same-named table of the other schema keeps all of its own
    assert await write_generation_triggers_on(ConnectionSession(second), _TABLE) == _ON
    assert await _functions(second, other) == {note_function_name(other, _TABLE)}


async def test_switching_off_twice_and_after_the_table_is_dropped_is_safe(
    two: tuple[asyncpg.Connection, asyncpg.Connection, str, str],
) -> None:
    first, _, schema, _ = two
    session = ConnectionSession(first)
    await switch_off(session, _TABLE)
    await switch_off(session, _TABLE)
    assert await write_generation_triggers_on(session, _TABLE) == frozenset()

    # switched on again, then the table dropped before its switch-off ran: the function is still taken away
    await give_switched_tables_their_trigger(session, [_TABLE])
    await give_noted_tables_their_note_trigger(session, [NotedTable(_TABLE, keys=("id",), tenancy=TENANT_BY_SCHEMA)])
    assert await _functions(first, schema) == {note_function_name(schema, _TABLE)}
    await first.execute(f"DROP TABLE {schema}.{_TABLE}")
    await switch_off(session, _TABLE)
    assert await _functions(first, schema) == set()
