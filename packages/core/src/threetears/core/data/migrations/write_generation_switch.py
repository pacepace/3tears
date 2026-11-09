"""a table switched off: the inverse of the runner's two hooks, inside the run (or upgrade) that turns it off.

A table switched on carries the write-mark trigger (:func:`~threetears.core.data.migrations.write_marks.give_switched_tables_their_trigger`)
and, when its key's text form is exact, its change-note function and triggers
(:func:`~threetears.core.data.migrations.change_notes.give_noted_tables_their_note_trigger`). A table its owner no
longer declares ``write_generation`` must lose them, or every write keeps queueing marks and notes for a table nothing
follows. :func:`switch_off` drops the write-mark trigger, every trigger running the table's note function, the shared
truncate trigger an earlier install gave, and the note function itself, each ``IF EXISTS``: safe to run twice, and on
a table already dropped (only the function is left to drop).

Runs on the caller's session, bound to the table's schema and holding the database-wide DDL lock (a migration run's,
or a data upgrade's); it takes no lock of its own.
"""

from __future__ import annotations

from typing import Any, Final

from threetears.core.ddl.change_notes import legacy_truncate_trigger_drop_sql, note_function_name
from threetears.core.ddl.write_marks import WRITE_MARK_TRIGGER
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["switch_off", "write_generation_triggers_on"]

log = get_logger(__name__)

_CURRENT_SCHEMA_SQL: Final = "SELECT current_schema() AS schema"

#: the triggers on one table of the current schema, with the function each runs
_TRIGGERS_SQL: Final = """
SELECT g.tgname AS trigger_name, p.proname AS function_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_proc p ON p.oid = g.tgfoid
WHERE n.nspname = current_schema() AND c.relname = $1 AND NOT g.tgisinternal
ORDER BY 1
"""

#: the schemas holding a function of the given name
_FUNCTION_SCHEMAS_SQL: Final = """
SELECT n.nspname AS schema FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname = $1 ORDER BY 1
"""

_TABLE_PRESENT_SQL: Final = """
SELECT 1 FROM information_schema.tables
WHERE table_schema = current_schema() AND table_name = $1 AND table_type = 'BASE TABLE'
"""


async def _here(store: Any) -> str:
    """the schema the session is bound to.

    :param store: the session's store
    :ptype store: Any
    :return: ``current_schema()``
    :rtype: str
    """
    return str((await store.query(_CURRENT_SCHEMA_SQL))[0]["schema"])


async def write_generation_triggers_on(store: Any, table_name: str) -> frozenset[str]:
    """the names of the write-generation triggers one table of the session's schema carries.

    :param store: a ``DataStore`` (or migration session) bound to the table's schema
    :ptype store: Any
    :param table_name: the table
    :ptype table_name: str
    :return: its write-mark trigger and every trigger running its note function, by name
    :rtype: frozenset[str]
    """
    function = note_function_name(await _here(store), table_name)
    return frozenset(
        str(row["trigger_name"])
        for row in await store.query(_TRIGGERS_SQL, table_name)
        if row["trigger_name"] == WRITE_MARK_TRIGGER or row["function_name"] == function
    )


async def switch_off(store: Any, table_name: str) -> None:
    """take the write-mark and change-note triggers off one table of the session's schema, and its note function.

    Safe to run twice, and on a table already dropped (only the function is left to drop).

    :param store: the run's ``DataStore`` (or migration session), bound to the table's schema and holding the DDL
        lock
    :ptype store: Any
    :param table_name: the table
    :ptype table_name: str
    :return: nothing
    :rtype: None
    """
    here = await _here(store)
    function = note_function_name(here, table_name)
    target = f"{quote_identifier(here)}.{quote_identifier(table_name)}"
    if await store.query(_TABLE_PRESENT_SQL, table_name):
        for trigger in sorted(await write_generation_triggers_on(store, table_name)):
            await store.execute(f"DROP TRIGGER IF EXISTS {quote_identifier(trigger)} ON {target}")
        # a table given the shared truncate trigger by an install before each table had its own
        await store.execute(legacy_truncate_trigger_drop_sql(table_schema=here, table=table_name))
    for row in await store.query(_FUNCTION_SCHEMAS_SQL, function):
        await store.execute(
            f"DROP FUNCTION IF EXISTS {quote_identifier(str(row['schema']))}.{quote_identifier(function)}()"
        )
    log.info(
        "table switched off: its write-mark and change-note triggers removed", extra={"extra_data": {"table": target}}
    )
