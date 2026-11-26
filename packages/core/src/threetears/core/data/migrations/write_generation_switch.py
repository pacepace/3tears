"""a table switched off: the inverse of the runner's two hooks, inside the run (or upgrade) that turns it off.

A table switched on before 0.95 may still carry the write-mark trigger. A table its owner no longer declares
``write_generation`` must lose it, or every write keeps queueing marks for a table nothing follows. :func:`switch_off`
drops it, ``IF EXISTS``: safe to run twice, and on a table already dropped. The change-note triggers and functions a
table carried until the ring retired are the cutover's to drop (:func:`~threetears.core.data.migrations.ring_cutover.
drop_ring`), never a switch's.

Runs on the caller's session, bound to the table's schema and holding the database-wide DDL lock (a migration run's,
or a data upgrade's); it takes no lock of its own.
"""

from __future__ import annotations

from typing import Any, Final

from threetears.core.ddl.write_marks import WRITE_MARK_TRIGGER
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["switch_off", "write_generation_triggers_on"]

log = get_logger(__name__)

_CURRENT_SCHEMA_SQL: Final = "SELECT current_schema() AS schema"

#: the triggers on one table of the current schema
_TRIGGERS_SQL: Final = """
SELECT g.tgname AS trigger_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = current_schema() AND c.relname = $1 AND NOT g.tgisinternal
ORDER BY 1
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
    :return: its write-mark trigger, when it carries one
    :rtype: frozenset[str]
    """
    return frozenset(
        str(row["trigger_name"])
        for row in await store.query(_TRIGGERS_SQL, table_name)
        if row["trigger_name"] == WRITE_MARK_TRIGGER
    )


async def switch_off(store: Any, table_name: str) -> None:
    """take the write-mark trigger off one table of the session's schema.

    Safe to run twice, and on a table already dropped.

    :param store: the run's ``DataStore`` (or migration session), bound to the table's schema and holding the DDL
        lock
    :ptype store: Any
    :param table_name: the table
    :ptype table_name: str
    :return: nothing
    :rtype: None
    """
    here = await _here(store)
    target = f"{quote_identifier(here)}.{quote_identifier(table_name)}"
    if await store.query(_TABLE_PRESENT_SQL, table_name):
        for trigger in sorted(await write_generation_triggers_on(store, table_name)):
            await store.execute(f"DROP TRIGGER IF EXISTS {quote_identifier(trigger)} ON {target}")
    log.info("table switched off: its write-mark trigger removed", extra={"extra_data": {"table": target}})
