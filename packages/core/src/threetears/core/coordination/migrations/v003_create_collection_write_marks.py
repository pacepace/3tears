"""coordination v003: the write-mark queue (receiver-trust Chunk 12), now only the empty table older replicas read.

Until 0.95 a switched-on table's trigger inserted a mark into ``collection_write_marks`` and a reconciler advanced the
table from it, and v003 created the queue and the trigger function. The marks are gone
(:mod:`threetears.core.ddl.write_marks`, coordination v006); v003 still creates the table, empty, so a database created
now holds what an older replica mid-roll reads, until the contract release drops it.

Each statement is idempotent and run on its own: YugabyteDB runs no DDL in a transaction.
"""

from __future__ import annotations

from threetears.core.ddl.write_marks import write_marks_ddl
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_collection_write_marks"]

log = get_logger(__name__)


async def create_collection_write_marks(store: DataStore) -> None:
    """create the write-mark queue in the bound schema (its trigger function is no longer created).

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    rows = await store.query("SELECT current_schema() AS schema")
    schema = str(rows[0]["schema"])
    log.info("creating the collection write-mark queue", extra={"extra_data": {"schema": schema}})
    for statement in write_marks_ddl(schema):
        await store.execute(statement)
