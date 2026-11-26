"""coordination v007: ``coordination_revocations`` carries its epoch, and a restore reads it a bucket at a time.

Append-only L3 (``design-append-only-l3.md`` decision 3, Pace 2026-11-25): the tables are their own change record, so
the change-note ring can be retired. The revocation list is the one coordination table switched on (its writes advance
a write generation, :func:`~threetears.core.coordination.migrations.register`), so the ring was its only restore path;
it is declared ``ROW_STAMPED`` (one row a key, carrying its last write's epoch, a delete or an expiry sweep marking it
gone in place). The other three coordination tables are not switched on and stay as they are.

The six steps of :func:`~threetears.core.data.migrations.stamp_tables.stamp_existing_table`: the epoch added nullable,
backfilled a page of keys a keyed ``UPDATE``, made ``NOT NULL``; the gone flag; the stored bucket, its
``(epoch_bucket ASC, epoch)`` index (split a tablet a bucket on YugabyteDB) and its ``COMMENT ON INDEX`` declaration.
On a table v001 created at this release the epoch and gone flag are already there (v001 renders the declaration) and
the backfill finds no row: only the bucket, the index and the comment are added.

Idempotent (each DDL ``IF NOT EXISTS``, the backfill only rows whose epoch is NULL), never truncating, for the
outage-window cutover. Forward-only.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.migrations.stamp_tables import StampingTable, stamp_existing_tables
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["revocations_carry_their_epoch"]

log = get_logger(__name__)

#: the table v007 stamps, its key as declared at v007 (FROZEN: a later change of its key is its own migration)
_STAMPED_TABLES: Final[tuple[StampingTable, ...]] = (
    StampingTable("coordination_revocations", (("purpose", "text"), ("key", "text")), tenancy=("schema", None)),
)


async def revocations_carry_their_epoch(store: DataStore) -> None:
    """stamp ``coordination_revocations``: its epoch, gone flag, restore bucket, index and declaration (v007).

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    stamped = await stamp_existing_tables(store, _STAMPED_TABLES)
    log.info("coordination v007: the revocation list is ROW_STAMPED", extra={"extra_data": {"rows_stamped": stamped}})
