"""coordination v004: the write-mark queue's ``(table_name, id)`` index.

The hub's broker reads, on every commit it answers, the marks of the tables the commit wrote, to clear those
its advances cover. With only the id key each read scanned the whole queue and the tombstones of every mark
cleared before (:func:`~threetears.core.ddl.write_marks.write_marks_index_sql`). A queue created by v003 from
this release on has the index already; this gives it to one created before.

Idempotent, one statement: YugabyteDB runs no DDL in a transaction, and builds the index online.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.core.ddl.write_marks import write_marks_index_sql
from threetears.observe import get_logger

__all__ = ["index_write_marks_by_table"]

log = get_logger(__name__)


async def index_write_marks_by_table(store: DataStore) -> None:
    """give the bound schema's write-mark queue its ``(table_name, id)`` index.

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    rows = await store.query("SELECT current_schema() AS schema")
    schema = str(rows[0]["schema"])
    log.info("indexing the collection write-mark queue by table", extra={"extra_data": {"schema": schema}})
    await store.execute(write_marks_index_sql(schema))
