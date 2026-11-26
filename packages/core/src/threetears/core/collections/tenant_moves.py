"""the moves table's declaration, and its cleanup: one row a stamped row's move between tenants (:mod:`~threetears.core.ddl.tenant_moves`).

A consumer that declares a stamped table a write may move between tenants (``TableSchema(moveable=...)``) declares this
table beside it, in the same schema, so it is created with it: its DDL is
``TENANT_MOVES_SCHEMA.create_statements(yugabyte=...)``. Its rows are appended by the write that moves the row, in the
same statement (``tenant_moves.with_moves_appended``), never by a collection of its own.

**Its key**: ``(table_name, scope, epoch, keys)``, ``scope`` the moved row's restore bucket: a restore reads one
table's bucket's moves since X as one range of it (on YugabyteDB the first column hashed, the rest ascending). The
table holds a row a move, and tenant moves are rare (a dataset promoted, a candidate refreshed), so one tablet a table
is no hot spot.

**Cleanup** (:func:`clear_moves`): a move stamped below the cleanup floor F is read by no restore, so quiet-time cleanup
deletes each table's moves below F a bucket at a time, one range of the key a statement, inside ``cleanup_deleting(F)``
as every cleanup delete is.
"""

from __future__ import annotations

from typing import Any, Final

from threetears.core.backends.broker_generation import cleanup_deleting
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, RESTORE_BUCKETS
from threetears.core.ddl.tenant_moves import TENANT_MOVES_TABLE
from threetears.core.eviction import Declared, Policy
from threetears.core.keyed import Fragment, delete_keyed
from threetears.core.tenancy import TENANT_BY_SCHEMA

__all__ = ["TENANT_MOVES_SCHEMA", "clear_moves"]

#: the moves table: the table a row of moved, the row's bucket, the move's epoch, the key, the tenant and the
#: invalidation columns it left (``tenant_moves``)
TENANT_MOVES_SCHEMA: Final = TableSchema(
    # the name and key spelled here, not by TENANT_MOVES_TABLE and its key constant: the no-whole-reads walk reads a
    # declaration's names only from its own module, and judges every read of the table by this key
    # (test_run_markers_schema holds them equal)
    name="threetears_tenant_moves",
    primary_key=("table_name", "scope", "epoch", "keys"),
    columns=[
        Column("table_name", STRING_TYPE),
        Column("scope", STRING_TYPE),
        Column("epoch", BIGINT_TYPE),
        Column("keys", STRING_TYPE),
        Column("tenant", STRING_TYPE, nullable=True),
        Column("columns", STRING_TYPE, nullable=True),
    ],
    # it lives among the owner's stamped tables, in the owner's own schema, so its tenancy is that schema
    tenancy=TENANT_BY_SCHEMA,
    l1_eviction=Declared(Policy.EVICTABLE, "a table's moves are read by key from L3; a cache of them may be dropped"),
)


async def clear_moves(pool: Any, schema: TableSchema, floor: int) -> int:
    """delete the moves of ``schema``'s table stamped below ``floor`` (the cleanup floor F), a bucket at a time: one
    range of the moves table's key ``(table_name, scope, epoch < F)`` a statement. Nothing for a table no write moves.

    :param pool: the L3 the table lives in (a pool, or the rail's proxy)
    :ptype pool: Any
    :param schema: the stamped table
    :ptype schema: TableSchema
    :param floor: F
    :ptype floor: int
    :return: the moves removed
    :rtype: int
    """
    if schema.move_record is None:
        return 0
    removed = 0
    below = Fragment(f'"{EPOCH_COLUMN}" < $1', (floor,))
    with cleanup_deleting(floor):
        for bucket in range(RESTORE_BUCKETS):
            written = await delete_keyed(
                pool, TENANT_MOVES_TABLE, by={"table_name": schema.name, "scope": str(bucket)}, where=[below]
            )
            removed += written.count
    return removed
