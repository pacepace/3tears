"""
agent-memory v031: the columns the collections declare NOT NULL are NOT NULL.

- ``customer_id`` on media, media_content and memory_chunks. The migrations
  built it nullable. It is set NOT NULL on each table that holds no NULL
  ``customer_id``; a table that does hold one is left as it is, with a WARNING
  naming the table and the count, because the missing value cannot be invented
  here and failing the whole migration would stop every package behind it.
- ``memories.date_updated``. v001 builds it NOT NULL, but a schema adopted from
  a consumer's own chain never ran v001 and can hold it nullable. A NULL takes
  the row's ``date_created`` (the row has not changed since, as far as anything
  recorded), then the column is set NOT NULL.

Replay guard: a column already NOT NULL is skipped, so a second run changes
nothing.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "CUSTOMER_ID_NOT_NULL_TABLES",
    "declared_not_null_columns",
]

log = get_logger(__name__)

#: The tables whose declaration has ``customer_id`` NOT NULL and whose
#: migrations built it nullable.
CUSTOMER_ID_NOT_NULL_TABLES = ("media", "media_content", "memory_chunks")


def _is_nullable_sql(table: str, column: str) -> str:
    """The catalog test for a nullable column in this schema.

    :param table: the table
    :ptype table: str
    :param column: the column
    :ptype column: str
    :return: an EXISTS expression
    :rtype: str
    """
    return f"""EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = '{table}'
           AND column_name = '{column}'
           AND is_nullable = 'YES'
    )"""


def _customer_id_not_null_sql(table: str) -> str:
    """The guarded ``SET NOT NULL`` on one table's ``customer_id``.

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
    IF {_is_nullable_sql(table, "customer_id")} THEN
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


_MEMORIES_DATE_UPDATED_NOT_NULL_SQL = f"""
DO $$
BEGIN
    IF {_is_nullable_sql("memories", "date_updated")} THEN
        UPDATE memories SET date_updated = date_created WHERE date_updated IS NULL;
        ALTER TABLE memories ALTER COLUMN date_updated SET NOT NULL;
    END IF;
END
$$
"""


async def declared_not_null_columns(store: DataStore) -> None:
    """Set the declared NOT NULL columns NOT NULL where the data allows.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("setting the declared NOT NULL columns NOT NULL (v031)")
    for table in CUSTOMER_ID_NOT_NULL_TABLES:
        await store.execute(_customer_id_not_null_sql(table))
    await store.execute(_MEMORIES_DATE_UPDATED_NOT_NULL_SQL)
