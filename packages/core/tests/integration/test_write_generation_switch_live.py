"""live: a table switched off loses the write-mark trigger a table switched on before 0.95 may carry, and nothing else.

Against a real database (the session ``db_container``). A schema holds coordination's migrations at platform scope, as
the platform schema does; a table there carries the write-mark trigger (put back by hand, as an older replica would)
and a trigger of its own. :func:`switch_off` drops the write-mark trigger ``IF EXISTS``, so the step runs twice and on
a dropped table. A same-named table in another schema keeps its own. The change-note triggers a table carried until the
ring retired are the cutover's (``ring_cutover.drop_ring``), never a switch's.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.coordination.migrations import register as register_coordination
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.migrations.session import ConnectionSession
from threetears.core.data.migrations.write_generation_switch import switch_off, write_generation_triggers_on
from threetears.core.ddl.write_marks import WRITE_MARK_TRIGGER

pytestmark = pytest.mark.integration

_TABLE = "switched_widgets"
_ON = {WRITE_MARK_TRIGGER}
#: a trigger of the table's own, which a switch-off never touches
_OWN = "switched_widgets_audit"


async def _schema(conn: asyncpg.Connection, schema: str) -> ConnectionSession:
    """``schema`` holding the table, carrying the write-mark trigger and one of its own, bound to the session."""
    await conn.execute(f"CREATE SCHEMA {schema}")
    await conn.execute(f"SET search_path TO {schema}")
    runner = MigrationRunner()
    register_coordination(runner)
    session = ConnectionSession(conn)
    await runner.apply_for_platform_schema(session)
    await conn.execute(f"CREATE TABLE {schema}.{_TABLE} (id TEXT PRIMARY KEY, n INTEGER)")
    await conn.execute(
        f"CREATE FUNCTION {schema}.switched_nothing() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END $$"
    )
    for trigger in (WRITE_MARK_TRIGGER, _OWN):
        await conn.execute(
            f'CREATE TRIGGER "{trigger}" AFTER INSERT ON {schema}.{_TABLE} '
            f"FOR EACH STATEMENT EXECUTE FUNCTION {schema}.switched_nothing()"
        )
    return session


async def _triggers(conn: asyncpg.Connection, schema: str) -> set[str]:
    rows = await conn.fetch(
        "SELECT g.tgname AS name FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2 AND NOT g.tgisinternal",
        schema,
        _TABLE,
    )
    return {str(row["name"]) for row in rows}


@pytest.fixture
async def two(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, asyncpg.Connection, str, str]]:
    """two schemas, each with the table carrying the write-mark trigger, each on a connection bound to it."""
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


async def test_switching_off_takes_the_write_mark_trigger_away_and_only_there(
    two: tuple[asyncpg.Connection, asyncpg.Connection, str, str],
) -> None:
    first, second, schema, other = two
    session = ConnectionSession(first)
    assert await write_generation_triggers_on(session, _TABLE) == _ON

    await switch_off(session, _TABLE)

    assert await write_generation_triggers_on(session, _TABLE) == frozenset()
    assert await _triggers(first, schema) == {_OWN}, "the table's own trigger stays"
    # the same-named table of the other schema keeps all of its own
    assert await write_generation_triggers_on(ConnectionSession(second), _TABLE) == _ON
    assert await _triggers(second, other) == {WRITE_MARK_TRIGGER, _OWN}


async def test_switching_off_twice_and_after_the_table_is_dropped_is_safe(
    two: tuple[asyncpg.Connection, asyncpg.Connection, str, str],
) -> None:
    first, _, schema, _ = two
    session = ConnectionSession(first)
    await switch_off(session, _TABLE)
    await switch_off(session, _TABLE)
    assert await write_generation_triggers_on(session, _TABLE) == frozenset()

    # the table dropped before its switch-off ran: the step still runs
    await first.execute(f"DROP TABLE {schema}.{_TABLE}")
    await switch_off(session, _TABLE)
