"""coordination v006: every write-mark trigger and the mark function taken down; the queue's table stays, empty.

A write to a switched-on table no longer leaves a mark (:mod:`threetears.core.ddl.write_marks`): a lost advance is
announced by its writer's backstop from the change notes (:mod:`threetears.core.collections.lost_writes`). This drops
every trigger named :data:`~threetears.core.ddl.write_marks.WRITE_MARK_TRIGGER` that calls the bound schema's mark
function -- on the bound schema's tables and on every agent schema's, whose triggers called the platform schema's --
then the function. The runner holds the database's DDL lock for the whole run.

**Rolling upgrade.** ``collection_write_marks`` stays until
:data:`~threetears.core.ddl.write_marks.WRITE_MARKS_CONTRACT_RELEASE`, so an older replica mid-roll reads an empty
queue rather than failing; triggers it puts back are taken down by the new replicas' backstop.

Idempotent, one statement at a time: YugabyteDB runs no DDL in a transaction.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.store import DataStore
from threetears.core.ddl.write_marks import MARK_FUNCTION, WRITE_MARK_TRIGGER
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["drop_write_mark_triggers"]

log = get_logger(__name__)

#: every write-mark trigger calling the bound schema's mark function, with its table
_MARK_TRIGGERS_SQL: Final = """
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_proc p ON p.oid = g.tgfoid
JOIN pg_namespace pn ON pn.oid = p.pronamespace
WHERE g.tgname = $1 AND NOT g.tgisinternal AND pn.nspname::text = current_schema()::text
ORDER BY 1, 2
"""


async def drop_write_mark_triggers(store: DataStore) -> None:
    """drop every write-mark trigger calling the bound schema's mark function, then the function.

    :param store: DataStore bound to the target schema via search_path, holding the run's DDL lock
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    rows = await store.query("SELECT current_schema() AS schema")
    schema = str(rows[0]["schema"])
    dropped: list[str] = []
    for row in await store.query(_MARK_TRIGGERS_SQL, WRITE_MARK_TRIGGER):
        table_schema, table = str(row["table_schema"]), str(row["table_name"])
        await store.execute(
            f"DROP TRIGGER IF EXISTS {quote_identifier(WRITE_MARK_TRIGGER)} "
            f"ON {quote_identifier(table_schema)}.{quote_identifier(table)}"
        )
        dropped.append(f"{table_schema}.{table}")
    await store.execute(f"DROP FUNCTION IF EXISTS {quote_identifier(schema)}.{quote_identifier(MARK_FUNCTION)}()")
    log.info(
        "write-mark triggers and their function taken down",
        extra={"extra_data": {"schema": schema, "tables": dropped}},
    )
