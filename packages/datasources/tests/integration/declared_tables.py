"""the datasources package's L3 tables as its declarations build them, on a test's own schema.

The integration tests that read or write ``datasources``, ``datasource_tables`` or ``datasource_customers`` hold
them to the shape the collections declare (:meth:`~threetears.core.collections.schema_backed.TableSchema.create_statements`:
every declared column, stamped tables' ``epoch``, ``gone`` and restore bucket, and the indexes the reads are keyed
by), never a hand-written copy, which drifted from the stamped tables and left a read of every declared column failing
on a column the copy never had. The hub's migrations build these tables in production; how they compare with the
declarations is :mod:`.test_collection_schema_parity_live`'s question, not these tests'. A row a test seeds by hand names its ``epoch`` and its dates: a stamped table's
epoch is ``NOT NULL`` with no default, and so are its dates.
"""

from __future__ import annotations

from typing import Any

from threetears.core.testing.plans import on_yugabyte

__all__ = ["create_declared"]


async def create_declared(conn: Any, *schemas: Any) -> None:
    """each of ``schemas`` created on ``conn``'s search path, with its indexes, as declared for the engine
    ``conn`` is (Postgres or YugabyteDB).

    :param conn: an asyncpg connection or pool, its session bound to the test's schema
    :ptype conn: Any
    :param schemas: the declared tables (``TableSchema``), parents first
    :ptype schemas: Any
    :return: nothing
    :rtype: None
    """
    yugabyte = await on_yugabyte(conn)
    for schema in schemas:
        for statement in schema.create_statements(yugabyte=yugabyte):
            await conn.execute(statement)
