"""coordination v005: each coordination table's expiry index range-sharded, built beside the hashed one it replaces.

The expiry sweep (``sweep_expired_sql``) reads a table's rows by ``expires_at < $1``. On YugabyteDB a secondary
index's first column is hashed unless declared ``ASC``, and a hashed index cannot answer a range: the sweep read
every tablet. Each table now declares ``idx_coordination_<table>_expiry_range`` on ``expires_at ASC`` (the same
btree on Postgres). A schema v001 creates from this release on has it already; this builds it beside the
``idx_coordination_<table>_expiry`` a schema created before carries, and only then drops the old one, so the sweep
always has an index to read.

Idempotent, one statement at a time: YugabyteDB runs no DDL in a transaction, and builds the index online.
"""

from __future__ import annotations

from threetears.core.coordination.tables import COORDINATION_TABLE_SCHEMAS, table_def_for
from threetears.core.data.sql_builder import build_create_index_sql
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["range_expiry_indexes"]

log = get_logger(__name__)


async def range_expiry_indexes(store: DataStore) -> None:
    """build each coordination table's range expiry index, then drop the hashed one it replaces.

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for schema in COORDINATION_TABLE_SCHEMAS:
        table = table_def_for(schema)
        log.info("range-sharding the coordination expiry index", extra={"extra_data": {"table": table.name}})
        for index in table.indexes:
            await store.execute(build_create_index_sql(table.name, index))
        await store.execute(f"DROP INDEX IF EXISTS idx_{table.name}_expiry")
