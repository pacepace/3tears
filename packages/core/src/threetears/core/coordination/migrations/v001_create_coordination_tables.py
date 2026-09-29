"""coordination v001: create the four coordination tables.

DDL is unqualified, so the caller's ``search_path`` decides the schema, and every statement is
idempotent so a replay on recovery is safe.

The statements are RENDERED from each collection's :class:`TableSchema` rather than written out
here, so the table a consumer migrates and the table the collection reads can never drift. A
consumer that cannot run DDL (an agent or tool pod, whose broker refuses it) declares the same
schemas in its data section instead; both paths read the one declaration.
"""

from __future__ import annotations

from threetears.core.coordination.tables import COORDINATION_TABLE_SCHEMAS, table_def_for
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_coordination_tables"]

log = get_logger(__name__)


async def create_coordination_tables(store: DataStore) -> None:
    """create the coordination counter, claim, revocation and redemption tables.

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for schema in COORDINATION_TABLE_SCHEMAS:
        log.info("creating coordination table", extra={"extra_data": {"table": schema.name}})
        table = table_def_for(schema)
        await store.execute(build_create_table_sql(table))
        for index in table.indexes:
            await store.execute(build_create_index_sql(table.name, index))
