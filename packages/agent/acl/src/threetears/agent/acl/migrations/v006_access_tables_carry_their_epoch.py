"""agent-acl v006: the five access tables carry their epoch, and a restore reads each a bucket at a time.

Append-only L3 (``design-append-only-l3.md`` decision 3, Pace 2026-11-25): the tables are their own change record, so
the change-note ring can be retired. ``namespaces``, ``groups``, ``group_members``, ``roles`` and ``role_assignments``
are switched on (each write advances its write generation, which every pod's access cache follows), so the ring was
their only restore path; each is declared ``ROW_STAMPED`` (one row a key, carrying its last write's epoch, a delete
marking its row gone in place: a group's memberships and grants with it, a namespace's grants with it, since the
database's ``ON DELETE CASCADE`` no longer runs for a row that is never deleted).

The six steps of :func:`~threetears.core.data.migrations.stamp_tables.stamp_existing_table` for each: the epoch added
nullable, backfilled a page of keys a keyed ``UPDATE``, made ``NOT NULL``; the gone flag; the stored bucket, its
``(epoch_bucket ASC, epoch)`` index (split a tablet a bucket on YugabyteDB) and its ``COMMENT ON INDEX`` declaration.

A deployment whose own migrations create these tables (the hub's v001) runs no agent_acl migration: it stamps them
with :data:`STAMPED_TABLES` through ``stamp_existing_tables`` in a migration of its own.

Idempotent (each DDL ``IF NOT EXISTS``, the backfill only rows whose epoch is NULL), never truncating, for the
outage-window cutover. Forward-only.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.migrations.stamp_tables import StampingTable, stamp_existing_tables
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["STAMPED_TABLES", "access_tables_carry_their_epoch"]

log = get_logger(__name__)

#: the tables v006 stamps, each key as declared at v006 (FROZEN: a later change of a key is its own migration)
STAMPED_TABLES: Final[tuple[StampingTable, ...]] = (
    StampingTable("namespaces", (("row_scope", "text"), ("namespace_id", "uuid"))),
    StampingTable("groups", (("row_scope", "text"), ("group_id", "uuid"))),
    StampingTable("group_members", (("group_id", "uuid"), ("id", "uuid"))),
    StampingTable("roles", (("role_id", "uuid"),)),
    StampingTable("role_assignments", (("row_scope", "text"), ("assignment_id", "uuid"))),
)


async def access_tables_carry_their_epoch(store: DataStore) -> None:
    """stamp the five access tables: each one's epoch, gone flag, restore bucket, index and declaration (v006).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    stamped = await stamp_existing_tables(store, STAMPED_TABLES)
    log.info("agent_acl v006: the access tables are ROW_STAMPED", extra={"extra_data": {"rows_stamped": stamped}})
