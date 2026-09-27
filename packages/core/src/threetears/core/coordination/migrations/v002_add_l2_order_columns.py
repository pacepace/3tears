"""coordination v002: give the compare-and-swap tables their L2 order columns, and backfill them.

``coordination_counters``, ``coordination_claims`` and ``coordination_redemptions`` are mutated
only through ``l2_cas_mutate``, which now persists every won swap to L3 fenced on the order it
won -- the L2 revision and the creation time of the stream it belongs to -- so an earlier winner
whose persist lands last cannot overwrite a later one. The order lives in two columns of the row
(``l2_epoch TIMESTAMPTZ``, ``l2_revision BIGINT``); a table without them is refused by
``l2_cas_mutate`` before it touches L2.

A table created by v001 on 0.55.0 or later already has them, because v001 renders the current
schema; every statement here is ``IF NOT EXISTS`` or ``WHERE ... IS NULL``, so it is a no-op there
and on a replay. A table created earlier gains both columns, and its existing rows are backfilled
to the order floor -- below every order a live bucket can produce -- so the first swap after this
migration supersedes them and every later one is fenced.

``coordination_revocations`` is not touched: it is written through ``save_entity``, never
compare-and-swapped, so it has no order to keep.

The DDL and the backfill are separate statements, run one at a time: YugabyteDB does not run DDL
and DML in one transaction.
"""

from __future__ import annotations

from threetears.core.collections.l2_order import l2_order_migration_statements
from threetears.core.coordination.tables import COORDINATION_TABLE_SCHEMAS
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_l2_order_columns"]

log = get_logger(__name__)


async def add_l2_order_columns(store: DataStore) -> None:
    """add ``l2_epoch`` / ``l2_revision`` to every compare-and-swap coordination table and backfill.

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for schema in COORDINATION_TABLE_SCHEMAS:
        if not schema.declares_l2_order:
            continue
        log.info("adding L2 order columns to coordination table", extra={"extra_data": {"table": schema.name}})
        for statement in l2_order_migration_statements(schema.name):
            await store.execute(statement)
