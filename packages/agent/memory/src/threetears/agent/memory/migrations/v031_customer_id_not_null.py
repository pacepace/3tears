"""
agent-memory v031: ``customer_id`` is NOT NULL on media, media_content and memory_chunks.

The collections declare the column NOT NULL on these three tables and the
migrations built it nullable, so a fresh schema disagreed with its own
declaration. This sets NOT NULL on each table that holds no NULL
``customer_id``. A table that does hold one is left as it is, with a
WARNING naming the table and the count: the missing value cannot be
invented here, and failing the whole migration would stop every package
behind it.

Replay guard: a column already NOT NULL is skipped, so a second run changes
nothing.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "CUSTOMER_ID_NOT_NULL_TABLES",
    "customer_id_not_null",
]

log = get_logger(__name__)

#: The tables whose declaration has ``customer_id`` NOT NULL and whose
#: migrations built it nullable.
CUSTOMER_ID_NOT_NULL_TABLES = ("media", "media_content", "memory_chunks")


def _set_not_null_sql(table: str) -> str:
    """The guarded ``SET NOT NULL`` for one table.

    :param table: the table
    :ptype table: str
    :return: a DO block
    :rtype: str
    """
    return f"""
DO $$
DECLARE
    missing bigint;
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = '{table}'
           AND column_name = 'customer_id'
           AND is_nullable = 'YES'
    ) THEN
        SELECT count(*) INTO missing FROM {table} WHERE customer_id IS NULL;
        IF missing = 0 THEN
            ALTER TABLE {table} ALTER COLUMN customer_id SET NOT NULL;
        ELSE
            RAISE WARNING 'v031: {table}.customer_id left nullable: % rows have no customer_id', missing;
        END IF;
    END IF;
END
$$
"""


async def customer_id_not_null(store: DataStore) -> None:
    """Set ``customer_id`` NOT NULL where no row lacks one.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("setting customer_id NOT NULL on %s (v031)", ", ".join(CUSTOMER_ID_NOT_NULL_TABLES))
    for table in CUSTOMER_ID_NOT_NULL_TABLES:
        await store.execute(_set_not_null_sql(table))
