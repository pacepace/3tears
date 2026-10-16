"""coordination v003: the write-mark queue and its trigger function (receiver-trust Chunk 12).

A switched-on table's trigger inserts a mark into ``collection_write_marks`` in the same transaction
as the write, and the reconciler advances the table's generation from it
(:mod:`threetears.core.collections.write_marks`). This creates the queue and the function in the
schema the runner binds, named by that schema so a trigger in any other schema writes the one queue.
The triggers themselves are installed by the reconciler, on every switched-on table in every schema,
because agent schemas are created after migrations run.

Each statement is idempotent and run on its own: YugabyteDB runs no DDL in a transaction.
"""

from __future__ import annotations

from threetears.core.collections.write_marks import write_marks_ddl
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_collection_write_marks"]

log = get_logger(__name__)


async def create_collection_write_marks(store: DataStore) -> None:
    """create the write-mark queue and its trigger function in the bound schema.

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
