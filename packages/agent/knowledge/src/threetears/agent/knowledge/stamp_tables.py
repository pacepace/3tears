"""the knowledge tables this package declares ``ROW_STAMPED``, as a migration that stamps them names them.

Append-only L3 (``design-append-only-l3.md`` decision 3, Pace 2026-11-25): the tables are their own change record, so
the change-note ring can be retired. ``concepts`` and ``playbook_entries`` are switched on (each write advances its
write generation, which every agent's knowledge scan cache follows), so the ring was their only restore path, and
``refuse_tables_without_epoch`` refuses every hub migration run until they carry an epoch. Each is declared
``ROW_STAMPED`` (:mod:`threetears.agent.knowledge.collections`): one row a key, edited in place by the hub a key at a
time, a delete marking its row gone in place (a shadow's origin link set NULL by 3tears with it, since the database's
``ON DELETE SET NULL`` no longer runs for a row that is never deleted).

This package has no migrations: the tables' DDL is the hub's (hub v012, v013, v014, v019, ...). The hub's migration
that stamps them calls :func:`knowledge_tables_carry_their_epoch` (the six steps of
:func:`~threetears.core.data.migrations.stamp_tables.stamp_existing_table` for each: the epoch added nullable,
backfilled a page of keys a keyed ``UPDATE``, made ``NOT NULL``; the gone flag; the stored bucket, its
``(epoch_bucket ASC, epoch)`` index and its ``COMMENT ON INDEX`` declaration), so the frozen keys and the steps live
beside the declarations they must match.

Idempotent (each DDL ``IF NOT EXISTS``, the backfill only rows whose epoch is NULL), never truncating, for the
outage-window cutover. Forward-only.
"""

from __future__ import annotations

from typing import Any, Final

from threetears.core.data.migrations.stamp_tables import StampingTable, stamp_existing_tables
from threetears.observe import get_logger

__all__ = ["STAMPED_TABLES", "knowledge_tables_carry_their_epoch"]

log = get_logger(__name__)

#: the tables a hub migration stamps, each key and tenant as declared at 0.95 (FROZEN: a later change of a key is its
#: own migration)
STAMPED_TABLES: Final[tuple[StampingTable, ...]] = (
    StampingTable("concepts", (("id", "uuid"),), tenancy=("column", "customer_id")),
    StampingTable("playbook_entries", (("id", "uuid"),), tenancy=("column", "customer_id")),
)


async def knowledge_tables_carry_their_epoch(store: Any) -> int:
    """stamp ``concepts`` and ``playbook_entries``: each one's epoch, gone flag, restore bucket, index and declaration.

    :param store: the hub migration's ``DataStore``, bound to the platform schema via search_path
    :ptype store: DataStore
    :return: the rows the backfills stamped
    :rtype: int
    """
    stamped = await stamp_existing_tables(store, STAMPED_TABLES)
    log.info("the knowledge tables are ROW_STAMPED", extra={"extra_data": {"rows_stamped": stamped}})
    return stamped
