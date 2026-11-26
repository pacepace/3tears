"""coordination v004: create the budget ledger's tables.

``coordination_ledger_accounts`` holds a :class:`~threetears.core.coordination.budget_ledger.BudgetLedger`'s
accounts, and ``coordination_documents`` its versioned records. Both are compare-and-swap tables, so
they are created with the L2 order columns already in place; v002 never sees them.

As in v001, the DDL is RENDERED from each collection's :class:`TableSchema`, so the table a consumer
migrates and the table the collection reads can never drift. It is unqualified, so the caller's
``search_path`` decides the schema, and each statement is ``IF NOT EXISTS`` and run on its own, so a
replay on recovery is safe and YugabyteDB, which runs no DDL in a transaction, takes it.
"""

from __future__ import annotations

from threetears.core.coordination.tables import LEDGER_TABLE_SCHEMAS, table_def_for
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_ledger_tables"]

log = get_logger(__name__)


async def create_ledger_tables(store: DataStore) -> None:
    """create the ledger accounts and documents tables.

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for schema in LEDGER_TABLE_SCHEMAS:
        log.info("creating coordination table", extra={"extra_data": {"table": schema.name}})
        table = table_def_for(schema)
        await store.execute(build_create_table_sql(table))
        for index in table.indexes:
            await store.execute(build_create_index_sql(table.name, index))
